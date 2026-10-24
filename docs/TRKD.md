# 时序图神经网络 → 轻量图神经网络的关系知识蒸馏：现状、方法与实证

> 本文档记录本轮方向调整（TGN 教师 → 1 层 GNN 学生）后的文献调研、方法设计、
> 实现、实验与结论。核心结论先行：
>
> 1. **稳健有效（相对监督）**：在修正时序邻居支撑（exact newest-K）后，
>    **CUT-FGW 关系 token OT + 时序关系注意力学生** 与 **候选级 logit+rank KD**
>    都显著优于监督学生：OT+kernel **+0.051（n=9）**、KD **+0.043（n=6）**。
>    但 **OT+kernel 相对 KD 仅 +0.008（n=9，t≈1.49，不显著）**：
>    OT 名义均值更高，KD 推理开销远低；二者统计上不可区分。
> 2. **不稳健/负向**：教师**邻居注意力质量**、**候选嵌入两两结构**、**源表示时序 drift**、
>    **候选成对偏好**、**支撑匹配弱教师**等新目标，均未能在多 seed 下稳定超过 KD；
>    其中注意力质量与教师正确率**负相关**，不适合作为蒸馏目标。
> 3. **工程修复**：PyG `LastNeighborLoader` 在单 batch 内节点度 > K 时会保留错误的“最近 K 条”；
>    已用 `ExactTemporalNeighborStore` 修复，并量化其对师生支撑一致率的影响（77% → 99.8%）。

---

## 1. 研究问题与范围

- 教师：连续时间时序 GNN（TGN：memory + 时间图注意力 + LinkPredictor）。
- 学生：轻量 GNN（1 层邻居聚合 + 残差 MLP），参数量 ~0.33M，推理不依赖 memory/多头注意力。
- 蒸馏范围：**时序维度上的关系知识**——节点/交互对象之间随时间变化的相对关系与关系演化；
  不引入以静态图拓扑（邻接、谱、传播矩阵）为核心的蒸馏目标。
- 评测：TGB 官方时间划分与历史负采样，官方 test MRR（协议不变）。

---

## 2. 文献调研（与本任务最接近的工作）

### 2.1 连续时间时序图与基准

| 工作 | 关系 |
|---|---|
| TGN (arXiv:2006.10637) | 教师结构；源节点表示 = 对历史邻居的注意力加权聚合 |
| TGB (arXiv:2307.01026) | 数据集/评测协议；官方负采样 |

### 2.2 图/关系知识蒸馏

| 工作 | 核心思想 | 与本文关系 |
|---|---|---|
| GLNN (arXiv:2110.08727) | GNN→MLP 输出 KD | 对应 `L_logit`；图无关基线 |
| LLP (arXiv:2210.05801) | 链路预测的关系蒸馏（anchor 为中心 rank/distribution matching） | 对应 `L_rank`；最接近的静态关系 KD |
| RKD (arXiv:1904.05068) | 匹配样本间距离/角度结构 | 启发候选结构蒸馏（TRSD） |
| Attention Transfer (arXiv:1612.03928) | 匹配教师注意力图 | 启发 TRID（时序注意力质量） |
| LightST (arXiv:2501.10459) | 时空 GNN→MLP 的时空 KD | 学生架构来源 |
| EasyST (arXiv:2409.06748) | 时空 GNN→MLP + 信息瓶颈 | 时空 KD 参考 |
| P&D (arXiv:2311.11759)、HKD (arXiv:2108.05507) | 显式结构/整体知识 | 结构 KD 参考 |

### 2.3 时序知识图谱蒸馏

MP-KD (arXiv:2303.14898)、LLM-TKG KD (arXiv:2602.14428, 2601.00202) 面向**离散快照 TKG**，
且常引入 LLM/额外教师；与连续时间事件流上的 TGN→轻量 GNN 不同。

### 2.4 空白

没有工作系统回答：**连续时间时序 GNN 的哪一类关系知识可以迁移到轻量 GNN？**
本文用受控实验比较了输出/排序、邻居注意力、候选结构、表示 drift、支撑匹配等目标。

---

## 3. 现有实现的不足（本轮修复/验证）

1. **邻居支撑错误（已修复）**：PyG `LastNeighborLoader.insert` 在单 batch 内节点度 > K 时，
   环形缓冲按位置覆盖写，随后 `topk` 保留的不一定是最近 K 条。tgbl-wiki 上 K=8 学生与
   K=24 教师的 event-id 支撑一致率仅 **77.0%**；修复后 **99.8%**（age 最大误差 3e-4）。
2. **关系槽旁支**：旧 CUT-FGW 的学生关系槽不进入打分路径，梯度只能经共享 trunk 间接影响 logits。
3. **注意力目标未验证**：此前默认教师注意力质量是“关系知识”，但未检验其与任务正确率的关系
   （本轮发现负相关，见 5.3）。
4. **评测方差**：学生 seed 方差大（KD 配置 std≈0.02–0.03），单/少 seed 结论不可靠；
   本轮统一 full val + 配对多 seed。

---

## 4. 方法与实现

### 4.1 精确时序邻居支撑（`ExactTemporalNeighborStore`）

对每个节点维护 `(neighbor, event_id)` 表；插入时把旧表与新事件合并，按 event id 排序取前 K。
保证：

```
S_s(t) = { 最近 K 条与 s 有关的交互 }，  与教师使用同一 event id 语义
```

教师侧同时输出 `relation_event_ids`，使师生支撑可**精确按 event id 对齐**，无需 OT 估计对应。

### 4.2 学生：1 层 GNN + 可选时序关系注意力

- `mean`：均匀聚合（旧行为）；
- `kernel`：单头加性注意力，输入 `[query 状态, 邻居状态, 相对 age]`，零初始化 ⇒ 初始均匀；
  权重直接乘邻居消息并进入打分路径。

### 4.3 蒸馏目标（可独立开关）

| 记号 | 目标 | 说明 |
|---|---|---|
| `L_logit` | KL(教师候选分布 ‖ 学生) | 候选级输出关系 |
| `L_rank` | 教师排序支撑上的一维 W1 | 候选级排序关系（LLP 式） |
| `L_OT` | CUT-FGW 关系 token 结构 OT | 旧方法；非平衡 Sinkhorn + FGW |
| `L_kernel` | 共享支撑上的注意力质量 KL | TRID；对齐教师邻居影响分布 |
| `L_struct` | 候选嵌入两两余弦结构 KL | TRSD；时间条件化相对相似性 |
| `L_drift` | 跨 batch 源表示一阶差分余弦 | TRDD；关系演化速度 |
| `L_pair` | 候选成对 soft RankNet | PRKD；成对偏好关系 |

所有新增项默认权重 0，`--method baseline` 行为不变（MLP 路径完全不变）。

### 4.4 理论分析

**命题 1（支撑一致性）**：若教师 soft target 是 `p_T(y | S_T(t))`，学生输入为 `S_S(t)`，
当 `S_S(t) ≠ S_T(t)` 时，KD 目标包含学生不可观测的信息，最小化 KD 不能保证学生逼近教师的
条件分布；支撑一致（`S_S = S_T`）是关系蒸馏可辨识的自然条件。

**命题 2（输出级关系的可辨识性）**：logit KD + rank KD 直接约束候选集合上的相对关系，
不要求师生表示空间对齐；这解释了为什么绝对嵌入 KD（`L_embed`）在表示空间不匹配时反而有害
（0.4653 vs KD 0.5247）。

**命题 3（注意力质量不是可靠目标）**：在 tgbl-wiki test（23,621 事件）上，
教师注意力熵与 MRR 的相关系数为 **+0.368**；按熵四分位，低熵组 MRR 0.416、高熵组 0.763；
在“成熟历史”（有效邻居 ≥15）子集内相关系数仍为 **+0.151**（四分位 0.594→0.800）。
即**注意力越集中，教师越容易错**；把注意力质量当作知识会传递错误信号。
（可能机制：集中注意力对应冷启动/歧义查询；即便控制历史长度，集中仍偏难。）

**命题 4（drift 的可迁移性有限）**：drift 方向在师生表示空间之间近似正交（训练中
`1-cos` 长期≈0.99），强对齐会与任务损失冲突；随机 drift 对照（0.5180）显著低于真实 drift
（0.5409，s42），说明方向含信息，但跨 seed 不稳定（n=6 配对 +0.006，t≈0.5）。

---

## 5. 实验

### 5.1 协议

- 数据：`tgbl-wiki`（110,232 train / 23,621 val / 23,621 test 事件；9,227 节点）。
- 教师：`checkpoints/wiki-tgn/teacher.pt`（TGN，test **0.658**；K=8/12/16/24 分别 0.616/0.637/0.652/0.658）。
- 学生：hidden 32、2 residual blocks、1 层邻居聚合、K=8、dropout 0.3、20 epochs、patience 5。
- 指标：官方 test MRR；val 用于早停/选模（full val），test 全量。
- 配对 seed 42–47；报告 mean±std 与配对 t。

### 5.2 主要正向结果（full val，n=6；n=9 复核）

| 方法 | seeds 42–47 | mean ± std（n=6） | 配对 Δ vs sup（n=6） |
|---|---|---|---|
| 监督（sup，mean） | 0.4688/0.4624/0.4682/0.4394/0.4180/0.4552 | 0.4520 ± 0.0199 | – |
| **KD（logit+rank，kernel）** | 0.5247/0.5017/0.4875/0.4503/0.5048/0.4991 | **0.4947 ± 0.0249** | **+0.0427 ± 0.0272（t≈3.85）** |
| **CUT-FGW OT + kernel** | 0.5479/0.5264/0.5061/0.4676/0.5059/0.4979 | **0.5086 ± 0.0271** | **+0.0566 ± 0.0240（t≈5.77）** |

**新 seed 复核（48–50，未参与方法选择）**：

| 方法 | s48 | s49 | s50 | mean ± std（n=9） |
|---|---|---|---|---|
| KD | 0.5052 | 0.4761 | 0.5086 | 0.4953 ± 0.0216 |
| OT + kernel | 0.4895 | 0.4956 | 0.4932 | 0.5033 ± 0.0229 |

- **OT+kernel 相对 KD（n=9）：+0.0080 ± 0.0161（t≈1.49，不显著）**。
  n=6 时的 +0.0139（t≈3.05）未能在新 seed 上复现；**不能宣称 OT+kernel 稳定优于 KD**。
- 两者相对监督都稳健为正：OT+kernel **+0.051（n=9 均值差）**，KD **+0.043（n=6）**。
- OT+kernel 的 val 均值与 KD 接近（0.596 vs 0.596），test 与 val 的增益方向不完全一致；
  跨 seed/跨数据集仍需谨慎。
- `OT + mean` 只有单 seed（s42 0.5045）；`OT + kernel` s42 0.5479，提示 kernel 聚合对 OT 有增益，
  但样本不足，不能单独下结论。
- 机制假设：kernel 学生把邻居聚合权重放到**打分路径**；OT 迁移教师关系 token 的二阶结构，
  二者共享 trunk，可能形成协同；但该协同的效应量 ~0.008，低于 seed 噪声，尚不能确认。

### 5.3 负向/不稳健结果

| 目标 | s42 | s46 | 结论 |
|---|---:|---:|---|
| TRID `kernel` w=1（K8，3000-val） | 0.5177 | – | 低于 KD（0.5247） |
| TRID w=0.25（full val） | 0.5152 | – | −0.0095 vs KD |
| TRID kernel-only（w=1） | 0.4706 | – | 低于 sup 0.4771 |
| TRSD 候选结构 w=0.5（n=3） | 0.5136 | – | 配对 +0.005（ns） |
| TRDD drift w=0.1（n=6） | 0.5409 | 0.4528 | +0.0060（t≈0.5） |
| TRDD drift-only | 0.4639 | – | 低于 sup |
| TRDD random drift | 0.5180 | – | 低于真实 drift |
| PRKD pr025 | 0.5301 | 0.5256 | s43 −0.0015，s44 −0.0138 |
| PRKD pr05nr | 0.5312 | 0.5162 | s43 −0.0098，s44 −0.0014 |
| SERCD 关系内容（KD + cos 0.5） | 0.5434 | 0.4994 | s42 +0.019，s46 −0.005；不稳健 |
| SERCD 关系内容（OT + cos 0.1/0.5） | 0.5175 / 0.4847 | 0.4762 | 低于 OT |
| 绝对嵌入 KD w=0.1 | 0.4653 | – | 显著有害 |
| 支撑匹配 K=8/12 教师 | 0.5211/0.5020 | 0.5076/0.5164 | 未超过 K=24 KD |

### 5.4 注意力价值诊断（teacher，test 23,621 事件）

| 指标 | 值 |
|---|---:|
| corr(attention entropy, MRR) | **+0.368** |
| corr(max attention mass, MRR) | −0.113 |
| corr(mass on newest 8, MRR) | −0.125 |
| MRR：熵最低四分位 → 最高四分位 | 0.416 → 0.763 |
| 成熟历史（≥15 邻居）内 corr | +0.151 |
| 成熟历史内 MRR：熵最低 → 最高四分位 | 0.594 → 0.800 |

→ 注意力集中不是“教师自信”，而是难例/冷启动；TRID 的负结果有机制解释。

---

## 6. 结论与建议

1. **相对监督稳健正向**：exact 支撑 + **候选级 logit+rank KD** 相对 sup **+0.043（n=6，t≈3.85）**；
   **CUT-FGW OT + 时序注意力学生** 相对 sup **+0.051（n=9 均值差）**。
2. **OT+kernel vs KD 不显著**：n=9 配对 +0.008（t≈1.49）；n=6 的 +0.014 未复现。
   OT 名义均值略高但计算贵；KD 是性价比最高的方案。**不能宣称新组合稳定优于 KD**。
3. **CUT-FGW OT 是有效的结构化关系蒸馏**；其收益依赖教师-学生余量
   （subreddit 上余量 ~0.006 时消失）。
4. **不要蒸馏教师邻居注意力质量**：它与正确率负相关（overall +0.368，
   成熟历史内 +0.151）；TRID 稳定为负/不显著。
5. **不要强对齐绝对表示或 drift**：表示空间不匹配；drift 方向近似正交，跨 seed 不稳定。
6. **工程上必须修正邻居支撑**：否则师生支撑一致率仅 77%，关系蒸馏的可辨识性受损。

### 下一步

- 跨数据集：subreddit 教师余量不足；若换 lastfm/review，需先确认教师显著强于学生。
- 若要继续寻找**稳定超过 KD** 的新目标，应优先选择**与教师正确率正相关**的关系信号
  （候选级 margin、正负边相对关系），并在蒸馏前做相关性诊断；避免直接使用注意力质量。
- 对 OT+kernel 的 +0.008 效应，需要 n≥20 或更大的效应量才能做出统计判断；当前证据不足。

---

## 7. 复现

```bash
# exact support + KD
python train.py --stage distill --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --student-arch gnn --student-aggregation mean --student-neighbors 8 \
  --student-neighbor-store exact --student-epochs 20 --patience 5 --dropout 0.3 \
  --structure-features --structure-feature-set basic \
  --logit-weight 0.5 --rank-weight 0.5 --relation-weight 0 \
  --kernel-weight 0 --diversity-weight 0 --student-seed 42

# CUT-FGW OT
python train.py --stage distill --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --student-arch gnn --student-aggregation mean --student-neighbors 8 \
  --student-neighbor-store exact --student-epochs 20 --patience 5 --dropout 0.3 \
  --structure-features --structure-feature-set basic \
  --method anchor_ot --anchor-weight 0 --relation-weight 0.5 --fused-alpha 0.20 \
  --logit-weight 0.5 --rank-weight 0.5 --diversity-weight 0.01 --ot-fast --student-seed 42
```
