# TRID：时序关系影响力蒸馏（Temporal Relational Influence Distillation）

> **状态（2026-10）**：本方法已实现并完成实验。结论是**负向/不显著**：
> 教师邻居注意力质量与教师正确率**负相关**（test 上 corr(entropy, MRR)=+0.368；
> 成熟历史内 +0.151），TRID 在 tgbl-wiki 上未超过 KD（w=0.25: 0.5152 vs 0.5247；
> kernel-only: 0.4706 vs sup 0.4771）。完整复评与最终方法见 [`TRKD.md`](TRKD.md)。
> 本文保留设计、数学与诊断过程，作为“什么关系知识不可迁移”的记录。

> 研究对象从“TGN → 图无关 MLP”调整为“**时序 GNN 教师（TGN）→ 轻量级 GNN 学生（1 层邻居聚合）**”。
> 蒸馏对象是**源节点在其历史交互上的条件影响力分布**（教师时序注意力的离散强度 / Hawkes 式影响核），
> 而不是静态图拓扑、节点特征或单纯输出 logits。

---

## 1. 研究现状与文献定位

### 1.1 连续时间时序图

- **TGN**（Rossi et al., arXiv:2006.10637）：memory + 时间图注意力 + 链接预测头；教师结构来源。
- **TGB**（Huang et al., arXiv:2307.01026）：官方时间划分与历史负采样，本文评测协议。

TGN 的源节点表示本质上是**对历史邻居的加权聚合**：

```
h_T(s, t) = Σ_{j ∈ H_s(t)} α_T(j | s, t) · φ_T(j) + self-term
```

其中 `α_T` 是时间图注意力在源节点邻居上的 softmax 质量，`φ_T(j)` 是边/邻居消息。
`α_T(·|s,t)` 随 `t` 与历史支撑 `H_s(t)` 演化，正是“时序关系知识”的载体。

### 1.2 图/关系知识蒸馏

- **GLNN**（arXiv:2110.08727）：GNN→MLP 的输出 KD。
- **LLP**（arXiv:2210.05801）：链路预测的**关系蒸馏**（以 anchor 节点为中心的 rank/distribution matching）。
- **RKD**（arXiv:1904.05068）：匹配样本间距离/角度结构，而非单样本输出。
- **Attention Transfer**（arXiv:1612.03928）：匹配教师注意力图。
- **LightST**（arXiv:2501.10459）/ **EasyST**（arXiv:2409.06748）：时空 GNN→MLP 的时空 KD。
- **P&D**（arXiv:2311.11759）、**HKD**（arXiv:2108.05507）：显式结构/整体知识蒸馏。

### 1.3 时序知识图谱蒸馏

- MP-KD（arXiv:2303.14898）、LLM-TKG KD（arXiv:2602.14428、2601.00202）主要面向**离散快照 TKG**，
  且常用 LLM/额外教师；与连续时间事件流上的 TGN→轻量 GNN 不同。

### 1.4 空白

据调研，**尚无**工作把连续时间时序 GNN 的**时序关系影响力算子**直接蒸馏进轻量 GNN，
并在**共享事件支撑**上做精确的分布匹配。现有图 KD 要么匹配输出/特征（GLNN、FitNet），
要么在静态图上匹配关系结构（LLP、RKD），要么用 OT 对齐无对应的 token 集合（CUT-FGW）。

---

## 2. 现有方法的不足（本项目已有证据）

1. **CUT-FGW 对齐的是“关系 token 的内容/二阶几何”**：教师 token 与学生固定槽之间没有真实对应，
   只能靠非平衡 Sinkhorn 近似；且学生关系槽不直接进入打分路径，梯度只能经共享 trunk 间接影响 logits。
   在 wiki 上 `rel=0.5` 有 +0.020，但在 subreddit 教师余量小（~0.006）时增益消失。
2. **Logit/排序蒸馏只约束输出**。给定输出 logits，注意力算子 `α_T` 并不可辨识：不同的邻居权重
   可产生相同排序。因此仅靠 KL/rank 无法保证学生学到教师的**时序聚合机制**。
3. **均匀 mean-pooling 是时间同质的**：它把最近 K 条交互等权平均，无法表达“哪些历史交互更重要、
   影响如何随 age 衰减/变化”。
4. **工程发现（已验证）**：PyG `LastNeighborLoader` 在单个 batch 内节点度 > K 时会用环形缓冲覆盖写，
   保留的并非“最近 K 条”。在 wiki 上，K=8 的学生与教师（K=24）的邻居支撑只有 **77%** 的槽位一致；
   这既影响旧 GNN baseline，也破坏“共享支撑”假设。本文用精确的 `ExactTemporalNeighborStore` 修复。

---

## 3. 关键问题

> **Q1**：时序关系蒸馏应该对齐什么对象？
> **Q2**：如何把该对象放进轻量 GNN 的**打分路径**，而不是一个旁支？
> **Q3**：教师/学生的事件支撑不同（K vs 24、hub 溢出）时如何精确对齐？

本文回答：对齐**条件影响力分布**（Q1），用**单头年龄/内容感知注意力**承接（Q2），
利用“师生观察同一事件流”的**共享支撑**做精确 lookup（Q3）。

---

## 4. 方法

### 4.1 教师侧：离散条件影响力

对源节点 `s`、查询时刻 `t`，教师邻居支撑 `H_s(t)` 为最近 `N` 条历史交互（按 event id）。
TGN 的注意力给出每条边的质量 `a_j`（多头平均）。归一化：

```
α_T(j | s, t) = a_j / Σ_{k ∈ H_s(t)} a_k ,   j ∈ H_s(t)
```

把它看作源节点历史事件上的**离散条件强度**：`α_T(j|s,t)` 是“过去事件 j 在时刻 t 对 s 的影响份额”。
这与 Hawkes 过程的激励核 `φ(t - t_j)` 同构（允许内容依赖），因此匹配 `α` 即匹配教师的时序关系过程。

**已验证的统计性质（tgbl-wiki，120 batch）**：

| 指标 | 值 | 含义 |
|---|---:|---|
| `entropy / log n` | 0.852 | 明显非均匀 |
| `max mass` | 0.296 | 存在主导交互 |
| `TV(uniform)` | 0.187 | 与均匀分布有实质差异 |
| `KL(α‖uniform)` | 0.153 | 同上 |
| `KL(α‖recency prior)` | 0.152 | **不能**用纯 recency 解释 |
| `corr(age, mass)` | −0.199 | 弱 recency，更多是内容依赖 |
| 最近 8 条质量占比 | 0.645 | K=8 只能覆盖约 2/3 质量 |

→ 教师影响力既非均匀也非纯时间衰减，具备可蒸馏的**关系内容**。

### 4.2 学生侧：轻量时序关系注意力

学生仍是 1 层邻居聚合，但把均匀均值换成**单头加性注意力**：

```
q_s = W_q (h_s + time(t))
k_j = W_k h_j + W_age · ρ(age_j)
e_j = v^T tanh(q_s + k_j)
w_j = softplus(e_j) · m_j / Σ_k softplus(e_k) · m_k
c_s = Σ_j w_j h_j
h_s' = h_s + g · MLP(c_s)
```

- `h_j` 是邻居节点的 base encoding（节点特征 + ID embedding）；
- `age_j = (t - t_j) / ΔT`，`ρ(age) = [r, r², r³]`，`r` 为行内归一化 age；
- `g` 为 0 初始化门控；`v` 零初始化 ⇒ 初始 `w` 为均匀分布，**不改变旧模型起点**；
- 参数仅 ~2k（hidden 32），远小于教师；仍是 1 层聚合、无 memory、无多头、无多层消息传递。

> 关键：注意力权重 `w` **直接乘在邻居消息上并进入打分路径**，与 CUT-FGW 的旁支关系槽不同。

### 4.3 蒸馏目标：共享支撑上的影响力匹配

因为师生重放**同一事件流**，学生的每条历史边都有全局 event id；教师在 `_relation_outputs` 中
同时输出 `relation_event_ids`。于是无需 OT 估计对应，直接按 event id 查表：

```
M = { (j, k) : eid_S[k] = eid_T[j] }
p_T[k] = α_T(eid_S[k]) / Σ_{k'∈M} α_T(eid_S[k'])     (在交集上重归一化)
p_S[k] = w_S[k] / Σ_{k'∈M} w_S[k']
L_TRID = E_s [ KL(p_T ‖ p_S) ]
```

- 交集外的学生边（教师因 K=24 截断或 hub 溢出未见）不惩罚；
- 教师质量在交集外（学生 K 更小）不强行要求，避免不可实现的支撑；
- 可选 `JS` 或 `MSE`（attention-transfer 对照），主方法用 `KL`。

**与 CUT-FGW 的区别**：

| | CUT-FGW | TRID |
|---|---|---|
| 对齐对象 | 关系 token 的**内容 + 二阶几何** | 历史交互上的**影响力分布** |
| 对应关系 | 未知，用非平衡 Sinkhorn/FGW 估计 | **已知**（同一 event id） |
| 学生侧 | 固定关系槽（旁支） | 注意力权重（**打分路径**） |
| 时间信息 | age 作为属性代价 | age 作为注意力输入 + 支撑排序 |
| 计算 | 22–24 ms/batch | 一次 lookup + KL（<1 ms） |

二者互补：TRID 迁移“谁重要/影响如何随 time 变化”，OT 迁移“关系表示的二阶结构”。

### 4.4 理论依据

**命题 1（算子误差界）**。设教师/学生对同一源节点的邻居消息为 `φ_j, ψ_j`，权重为 `α_j, w_j`，
`‖φ_j‖,‖ψ_j‖ ≤ B`，`‖φ_j − ψ_j‖ ≤ ε`。则

```
‖h_T − h_S‖ ≤ ε + 2B · TV(α, w)
            ≤ ε + B · sqrt(2 · KL(α ‖ w))
```

第二式由 Pinsker 不等式。因此最小化 `KL(α‖w)` 直接约束教师-学生**聚合表示**的差异；
而仅匹配 logits 不提供该约束。

**命题 2（logits 的非可辨识性）**。链接预测头对源表示是线性的（TGN 的 LinkPredictor）：
`logit(c) = u^T (h_s ⊙ h_c)`。给定 `h_s` 的输出，存在多组不同的邻居权重 `α` 产生同一 `h_s`
（只要加权和相同）。因此 logit KD 的可行解集合严格大于 `α` 匹配的可行解集合；
显式影响力匹配提供了额外的函数空间约束。

**命题 3（Hawkes 解释）**。把 `α_T(j|s,t)` 视为历史事件的条件强度份额，
`w_S(j|s,t)` 是学生参数化的核；最小化 KL 等价于最小化该条件强度过程与教师过程的
前向 KL，故学到的是**关系影响的时序演化规律**（不同 age/内容下的相对激励），
而非静态邻接结构。

### 4.5 为什么不是“堆损失”

- 只有**一个**新监督项（共享支撑 KL），其余 `CE/logit/rank` 是已有 baseline；
- 该项的监督信号来自教师的**实际聚合算子**（不是人工先验），且与 CE/logit 在函数空间上
  约束不同的对象；
- 学生结构只增加一个**可解释的最小注意力头**（零初始化、均匀起点），没有 memory/多头/多层；
- 所有消融都可拆开：`kernel-weight=0` 即“同架构仅监督”、`logit/rank=0` 即“只有影响力蒸馏”。

---

## 5. Baseline 与实验协议

### 5.1 模型 baseline

| 名称 | 说明 |
|---|---|
| Teacher (TGN) | 冻结，wiki test 0.6806 |
| Sup-Mean | 1 层 GNN，均匀均值（旧行为 + 精确邻居存储） |
| Sup-Kernel | 同架构、可学习时序注意力，**无 KD**（隔离架构收益） |

### 5.2 蒸馏 baseline

| 名称 | 说明 |
|---|---|
| KD (logit+rank) | GLNN/LLP 式输出/排序蒸馏 |
| OT (CUT-FGW) | 本项目旧方法，`rel=0.5, fa=0.2, div=0.01` |
| AT-MSE | 同一注意力架构，但用 MSE 匹配教师权重（attention transfer 对照） |
| **TRID (ours)** | 共享支撑 KL |

### 5.3 设置

- 数据：`tgbl-wiki`（主）、`tgbl-subreddit`（若 wiki 正向）。
- 指标：TGB 官方 test MRR；val 用 `--eval-max-events 3000` 仅做早停（test 全量）。
- 学生：hidden 32、2 residual blocks、K=8/24、dropout 0.3、20 epochs、patience 5。
- 教师、数据划分、负采样、训练预算完全一致；配对 seed 42/43/44（必要时 n=6）。
- 公平性：Sup-Kernel 与 TRID 使用**完全相同**的学生架构和参数量，仅 KD 项不同。

---

## 6. 实验记录

### 6.1 已完成的诊断（tgbl-wiki）

- 教师影响力非均匀、非纯 recency（见 4.1 表）。
- 修复邻居存储后，师生支撑 event-id 一致率 **99.8%**，age 最大误差 `3e-4`。
- K=8 覆盖教师质量 **48.7%**（按 24 槽归一化）；最近 8 槽覆盖 **64.5%**。
  → 主实验同时测试 K=8（轻量）与 K=24（全覆盖）。

### 6.2 Round-1（seed 42）与 Round-2（full val）

- s42，kernel w=0.25 + logit/rank：0.5152（KD 0.5247）。
- s42，kernel-only w=1.0：0.4706（sup 0.4771）。
- K=24，kernel w=1.0（3000-val）：0.4955。
- 结论：**TRID 不优于 KD，且在部分配置下低于监督**；见 `RESULTS.md` 表 25。

### 6.3 待验证假设

- H1：`Sup-Kernel` ≥ `Sup-Mean`（时序注意力本身有效）。
- H2：`TRID` > `Sup-Kernel`（影响力蒸馏带来额外收益，而非架构）。
- H3：`TRID` 与 logit/rank 可叠加；与 OT 互补。
- H4：K=24 时 TRID 收益更大（教师质量覆盖更完整）。

### 6.4 已知失败/风险

- `kernel-weight` 过大可能压过 CE（需扫描 0.5/1/2）。
- 教师注意力可能包含与任务无关的噪声；若 TRID 无收益，需检查
  “打乱教师 mass”的 control（随机目标）以确认信号来源。
- subreddit 教师余量仅 ~0.006，TRID 也可能无增益；这属于数据上限而非方法失败。

---

## 7. 与已有工作的区别（一句话）

> 现有图/关系 KD 匹配输出、特征或静态关系结构；TRID 首次把**连续时间时序 GNN 的邻居注意力**
> 解释为历史事件上的条件影响强度，并在**共享事件支撑**上把该**时序关系影响算子**蒸馏进
> 轻量 GNN 的打分路径。

---

## 8. 复现命令

```bash
# 诊断教师影响力
python diagnose_temporal_kernel.py --batches 120

# Sup-Kernel（架构控制）
python train.py --stage student --dataset tgbl-wiki --student-arch gnn \
  --student-aggregation kernel --student-neighbors 8 \
  --student-epochs 20 --patience 5 --dropout 0.3 \
  --structure-features --structure-feature-set basic --student-seed 42

# TRID
python train.py --stage distill --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --student-arch gnn --student-aggregation kernel --student-neighbors 8 \
  --student-epochs 20 --patience 5 --dropout 0.3 \
  --structure-features --structure-feature-set basic \
  --logit-weight 0.5 --rank-weight 0.5 --relation-weight 0 \
  --kernel-weight 1.0 --kernel-divergence kl --student-seed 42
```
