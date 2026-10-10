# IDEA：Anchor Relation + Structural OT 时序图知识蒸馏

> 本文档描述在当前 CUTFGWD 项目上实现的统一蒸馏方法。所有设计均对应到真实代码，
> 未引入代码中不存在的字段或接口。

---

## 1. 方法背景

时序图神经网络（Temporal GNN，本文以 TGN 为代表）通过 **记忆模块 + 时间图注意力**
在连续时间事件流上学习节点表示，在动态链路预测上显著优于静态图模型。但其推理需要
多跳历史邻居，延迟高、依赖图结构，难以部署在低延迟或图结构不可得的场景。

知识蒸馏（KD）是把大模型能力迁移到轻量模型的标准手段。对链路预测而言，直接将
**图无关 MLP**（只吃节点特征、ID、时间）作为学生，可以把推理从"图依赖"变成"纯特征
前向"，但会丢失关系（relational）知识。本方法的目标是：在把 TGN 蒸馏到 MLP 的同时，
**显式迁移教师的关系表示与关系结构**。

---

## 2. 现有方法存在的问题

1. **教师关系头从未被训练**：原 `train_teacher_epoch` 始终以 `return_relations=False`
   调用教师，`PyGTGNTeacher.relation_projector` 拿不到任何梯度，是随机初始化。此时
   基于关系的蒸馏（CUT-FGW）是在对齐随机投影，毫无意义。
2. **零权重分支仍被计算**：原 `CUTFGWDDistillation.forward` 无条件计算 KL、rank、
   CUT-FGW、diversity，即便对应权重为 0，浪费且掩盖配置。
3. **学生关系槽不进入打分路径**：`LightSTMLPStudent` 的 `relation_tokens` 不参与
   `logits`，因此关系知识只能通过共享 trunk 间接影响排序。
4. **缺少独立学生种子**：教师与学生共用 `--seed`，无法做配对随机种子实验。
5. **教师/学生 `relation_dim` 未校验**：维度不一致会在 CUT-FGW 张量运算时崩溃。

---

## 3. 核心 Idea

统一为 **Anchor Relation + Structural OT**：

- **Anchor Relation（年龄锚定关系蒸馏）**：教师的历史关系 token 与学生的固定关系槽
  都按"年龄（越新越小）"排序；用年龄做锚点建立最近邻对应，再在表示空间对齐。这样
  变长教师轨迹与固定学生槽之间获得稳定对应，迁移的是**关系表示内容**。
- **Structural OT（结构最优传输蒸馏）**：复用现有 CUT-FGW，把教师关系轨迹的**二阶
  结构**（token 间余弦关系矩阵 + 相邻时间差分关系矩阵）通过非平衡 Sinkhorn + FGW
  传输到学生固定槽，迁移的是**关系结构**。
- **Prediction Distillation**：KL 对齐候选预测分布（Pointwise）。
- **TARD 排序关系**：复用 `rank_wasserstein_loss`，对齐候选排序结构（排序关系）。
- 前置修复：给教师增加自监督关系头，使关系表示"可靠"，再谈关系蒸馏。

---

## 4. Teacher 和 Student 的具体结构

### Teacher：`PyGTGNTeacher`（`model/teacher.py`）

- `TGNMemory`（IdentityMessage + LastAggregator）：维护节点记忆 `memory`、`last_update`。
- `GraphAttentionEmbedding`（堆叠 `TransformerConv`）：基于历史消息生成时间图嵌入。
- `LinkPredictor`：`source_embedding` 与 `candidate_embedding` → 候选 logits。
- `relation_projector`：把 `[neighbor_memory || edge_attributes]` 投影为 `relation_dim`
  的历史关系 token。
- **`relation_align_head`（固定组件）**：`Linear(relation_dim -> hidden_dim)`，始终构建，
  用于教师预训练阶段的自监督关系对齐。`--teacher-relation-weight` 只控制该辅助项权重
  （`baseline` 默认 0，即不训练该头），且该头**不参与打分路径**。

教师在 `score_candidates(..., return_relations=True)` 时额外返回 `source_embedding`
（打分用的源节点嵌入），仅用于关系头自监督，不影响打分路径。

### Student：`LightSTMLPStudent`（`model/student.py`）

- 图无关：只读取 `src`、`candidates`、`timestamp`。
- `node_encoder`（MLP over 节点特征）+ `node_embeddings` + `TimeEncoding` +
  若干 `ResidualMLPBlock` + `score_head` → `logits[B, C]`。
- 关系分支：`relation_slots` 个固定槽，槽年龄由可学习 `age_gaps` 生成
  （`_ordered_slot_ages`，从旧到新单调），`relation_decoder` 产出
  `relation_tokens[B, R, relation_dim]`，`relation_mass_head` 产出槽质量 logits。
- **学生结构本次不改**，保持轻量。

---

## 5. Anchor Relation 的定义

教师关系 token 记为 `{z^T_j}`，带年龄 `{a^T_j}` 与 mask；学生槽记为 `{z^S_k}`，
年龄 `{a^S_k}`（均按事件时间归一化，越新越小）。

**年龄锚点**：对每个学生槽 `k`，在有效教师 token 中找年龄最近者

```
j*(k) = argmin_{j: mask_j=1} | a^S_k - a^T_j |
```

**Anchor Relation Distillation**（`loss/kd.py: anchor_relation_loss`）：

```
L_anchor = mean_{k: 该样本有有效教师 token}
           [ 1 - cos(z^S_k, sg(z^T_{j*(k)})) + || z^S_k - sg(z^T_{j*(k)}) ||^2 ]
```

`sg` 为 stop-gradient（教师侧 detach）。cos 对齐方向、MSE 对齐尺度。

---

## 6. Structural OT / CUT-FGW 的作用

`loss/cutfgw.py: CausalUnbalancedTemporalFGW`（**原样复用**）输入：

- `teacher_tokens[B, N, relation_dim]`、`teacher_ages[B, N]`、`teacher_mask[B, N]`、
  `teacher_mass[B, N]`（缺失时回退 recency 先验）；
- `student_tokens[B, R, relation_dim]`、`student_ages[B, R]`、
  `student_mass_logits[B, R]`。

构造两组二阶关系：`memory_relation`（token 间余弦距离矩阵）与
`dynamic_relation`（相邻 token 差分后的关系矩阵），按 `fused_alpha` 融合属性代价与
结构代价，用非平衡 Sinkhorn 迭代求 transport plan，最后以 plan 加权代价得到
`L_OT`（`relation`）。它负责对齐**关系的结构**，与 Anchor Relation 的**逐槽表示对齐**
互补。

---

## 7. 完整蒸馏流程

1. **教师预训练**（`pretrain_teacher`）：CE 训练；若 `--teacher-relation-weight > 0`，
   同时请求 `return_relations=True` 并叠加关系头自监督对齐，使 `relation_projector`
   被训练。
2. **冻结教师**（`freeze_teacher`）。
3. **学生初始化**：用独立 `student_seed` 重置随机源后构建。
4. **每个 epoch**：
   - `teacher.reset_state()`；按需请求教师/学生关系输出；
   - 教师 `score_candidates` → `update_state`（官方 TGN 顺序：先打分后写正边）；
   - 学生前向；
   - `CUTFGWDDistillation` 计算总损失并反向；
   - 在官方验证集上早停/选最佳。
5. **保存学生 checkpoint，官方测试集评测（TGB MRR）**。

---

## 8. 数学公式

记 `L_sup` 为候选交叉熵。总损失：

```
L = w_task · L_sup
  + w_logit · KL( softmax(z_T/τ) || softmax(z_S/τ) ) · τ^2      (Prediction)
  + w_rank  · W1_teacher-order(z_S, z_T)                         (TARD 排序关系)
  + w_anchor· L_anchor                                            (Anchor Relation)
  + w_relation · L_OT                                             (Structural OT)
  + w_diversity · L_div                                           (关系槽多样性)
```

教师关系头自监督：

```
L_align = mean_{有效行} [ 1 - cos( A(mean_j z^T_j), sg(h_s) )
                          + || A(mean_j z^T_j) - sg(h_s) ||^2 ]
```

其中 `A = relation_align_head`，`h_s = source_embedding`，`mean_j` 为 mask 加权均值。

---

## 9. 总体 Loss 与实现位置

- 总损失类：`loss/distillation.py: CUTFGWDDistillation`，新增 `anchor_weight` 参数。
- 各分量：
  - `L_sup`：`F.cross_entropy`；
  - `L_pred`：`loss/kd.py: logit_kd_loss`；
  - `L_rank`：`loss/kd.py: rank_wasserstein_loss`；
  - `L_anchor`：`loss/kd.py: anchor_relation_loss`（新增）；
  - `L_OT`：`loss/cutfgw.py: CausalUnbalancedTemporalFGW`；
  - `L_div`：`loss/kd.py: relation_slot_diversity_loss`；
  - `L_align`：`loss/kd.py: relation_alignment_loss` + `masked_relation_mean`（新增），
    在 `train.py: train_teacher_epoch` 中调用。
- 权重为 0 的分量在 `forward` 中**真正跳过**，并返回设备/dtype 一致的零标量，日志字段
  仍保留：`total, task, logit, rank, anchor, relation, diversity`。

---

## 10. 与普通 Knowledge Distillation 的区别

| 维度 | 普通 KD | 本方法 |
|---|---|---|
| 迁移对象 | 单样本输出 logits | logits + 候选排序关系 + **关系表示** + **关系结构** |
| 学生图依赖 | 常仍依赖图 | 完全图无关 MLP |
| 时序性 | 无 | 年龄锚定、时间差分结构 |
| 教师关系头 | 不涉及 | 显式自监督训练后再蒸馏 |

---

## 11. 与原 CUT-FGW 方法的区别

| 维度 | 原 CUT-FGW | 本方法 |
|---|---|---|
| 教师关系头 | 随机、未训练 | 通过 `L_align` 训练 |
| 零权重分支 | 全算 | 真正跳过 |
| 关系对齐层次 | 仅结构（OT） | 结构（OT）+ 逐槽表示（Anchor） |
| 学生种子 | 与教师共用 | 独立 `--student-seed` |
| 可切换性 | 固定 | `--method baseline\|anchor_ot` |

---

## 12. 为什么这样设计

- 关系蒸馏的前提是教师关系表示可靠，因此先修教师关系头（`L_align`）。
- 学生槽数量固定、教师历史变长，直接逐 token 对齐不可行；两者都天然按年龄排序，
  因此用**年龄**作为锚点建立对应，既稳定又无需额外参数。
- 结构与内容分开：OT 对齐二阶结构，Anchor 对齐一阶表示，二者互补、不冗余。
- 全部改动以"新增参数 + 可选分支"实现，默认关闭时与旧行为完全一致。

---

## 13. 创新点

1. **年龄锚定的关系表示蒸馏**：用时间年龄在变长轨迹与固定槽间建立无需学习的稳定对应。
2. **Anchor Relation + Structural OT 统一框架**：一阶表示对齐与二阶结构 OT 互补。
3. **可靠教师关系头**：以打分的 `source_embedding` 为自监督目标训练关系投影。
4. **零开销 baseline 兼容**：`--method` 切换，默认路径数学上与原实现一致。

---

## 14. Baseline 设置

**模型 baseline（模型对比）**

- TGN 教师（`--stage teacher`），官方 test MRR。
- 监督 MLP 学生（`--stage student`），无蒸馏。
- （可选）静态 GNN→MLP 方法如 GLNN 的思路作为外部参考。

**蒸馏方法 baseline（方法对比）**

- 纯 logit KD（`--logit-weight > 0`，其余 0）。
- 原 CUT-FGW 联合目标（`--method baseline`，`teacher_relation_weight=0`）。
- 本方法（`--method anchor_ot`）。

> 对比维度：同一数据集/划分/官方负采样协议下的官方 test MRR、参数量、推理延迟。

---

## 15. 数据集与实验设置

- 数据集：TGB（`tgbl-wiki`、`tgbl-coin` 等），官方时间划分与官方负采样。
- 指标：官方 MRR（`evaluate_tgb`）。
- 公平性：`--seed` 固定数据与负采样；`--student-seed` 取 42/43/44 做配对比较。
- 早停：按官方验证 MRR，`--patience`。
- 所有 baseline 与新方法使用同一数据签名与同一教师 checkpoint。

---

## 16. Ablation Study 设计

1. `--teacher-relation-weight ∈ {0, 1}`：关系头是否训练。
2. `--anchor-weight ∈ {0, 0.1, 0.5, 1.0}`：Anchor 关系蒸馏强度。
3. `--relation-weight ∈ {0, 1}`：Structural OT 是否开启。
4. `--logit-weight` / `--rank-weight` 单独开关：Prediction 与排序关系贡献。
5. 完整 vs 去 Anchor vs 去 OT：验证互补性。
6. 教师关系头目标消融：`L_align` vs 不使用（回到随机关系头）。

---

## 17. 预期实验结果

- 在 synthetic smoke 上：数据→教师→学生→关系→CUT-FGW→loss→backward→step
  全流程跑通，无 NaN/Inf（已在本机 DGL 环境验证）。
- 在 TGB 上：预期 `anchor_ot` 的官方 test MRR ≥ 纯 logit KD ≥ 原 CUT-FGW；
  Anchor 与 OT 同时开启应优于任一单开。
- **说明**：当前工作区缺少 TGB 数据与 `py-tgb`，正式的 `tgbl-wiki` 数值需在具备
  数据与 CUDA 的 Ubuntu/项目副本上运行后填入。

---

## 18. 方法整体框架图（文字描述）

```
               事件流 batch: {src, candidates, timestamp, event_time, edge_features, ...}
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
   [Teacher: PyGTGNTeacher]             [Student: LightSTMLPStudent]
   memory + temporal attention           node feat + id + time
   + LinkPredictor                       + residual MLP + score_head
   + relation_projector                  + relation slots
   relation_align_head                        │
        │                                     │
   logits_T[B,C]                          logits_S[B,C]
   relation_tokens_T[B,N,rd]              relation_tokens_S[B,R,rd]
   relation_ages_T[B,N]                   relation_ages_S[B,R]
   relation_mask_T[B,N]                   relation_mass_logits_S[B,R]
   source_embedding[B,h]                       │
        │                                     │
        ├── (教师预训练) L_align: A(pool(z_T)) ↔ sg(source_embedding)
        │
        └──────────────► CUTFGWDDistillation ◄──────────────┘
                 │        │         │            │
             L_pred    L_rank   L_anchor      L_OT (CUT-FGW)
              (KL)     (W1)   (年龄锚定对齐)  (非平衡Sinkhorn+FGW)
                 └────────┴─────────┴────────────┘
                          │
                 L = w_task·L_sup + w_logit·L_pred + w_rank·L_rank
                     + w_anchor·L_anchor + w_relation·L_OT + w_div·L_div
                          │
                       backward → optimizer.step()
```

---

## 19. 后续可扩展方向

1. **Anchor 关系用软对应**：把最近邻替换为按年龄的核加权/插值，缓解边界槽。
2. **分布级关系蒸馏**：在 Anchor 上加入二阶矩（协方差）匹配，增强表示迁移。
3. **动态负采样**：训练候选随 epoch 变化（需在 `CandidateBatchLoader` 增加
   `set_epoch`），扩大覆盖但会改变协议，需单列实验。
4. **因果干预关系蒸馏**：对历史邻居做 do-操作，蒸馏教师打分的因果差分。
5. **结构化先验**：用 teacher_mass 作为 OT 质量先验的进一步校准。
6. **小 GNN 学生**：在保持 graph-free 之外，对比轻量 GNN 学生的权衡。

---

## 附：运行方式

```bash
# 环境（本机验证）：D:\Anaconda\envs\DGL\python.exe （torch 2.1.0 + pyg 2.6.1）

# baseline（与旧行为一致）
python train.py --stage all --dataset tgbl-wiki --output-dir checkpoints/wiki-tgn

# 原 CUT-FGW 蒸馏 baseline
python train.py --stage distill --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --method baseline --output-dir checkpoints/wiki-kd-baseline

# 新方法 Anchor Relation + Structural OT
python train.py --stage distill --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --method anchor_ot --anchor-weight 0.5 --relation-weight 1.0 \
  --logit-weight 0.1 --rank-weight 0 --diversity-weight 0 \
  --student-seed 42 --output-dir checkpoints/wiki-anchor-ot-s42
```

---

## 20. 实测诊断与工程修复

### 20.1 参数量对比的正确解读（tgbl-wiki, num_nodes=9227）

| 模型 | 可训练参数 | 组成 | 运行时状态 |
|---|---:|---|---:|
| Teacher (TGN hidden=32, 2 层) | 68,417 | memory 27,488 + GNN 20,512 + relation_projector 18,304 + link 2,145 | 775,071 |
| Student (MLP + node embedding) | 323,818 | node_embeddings 295,264 (91%) + 其余 ~28k | 0 |

教师没有 per-node 参数（信息在运行时 memory 中）；学生的参数几乎全是随节点数增长的
查找表。二者不可直接比较，应同时报告参数与运行时状态。

### 20.2 蒸馏耗时分解（实测 batch=128）

| 阶段 | 修复前 | 修复后 |
|---|---:|---:|
| teacher 前向（无关系） | 2.58 ms | 2.45 ms |
| teacher 关系抽取增量 | 36.77 ms | 1.17 ms |
| student 前向 | 1.58 ms | 1.64 ms |
| CUT-FGW 目标 | 22.29 ms | 23.93 ms |

瓶颈原是 `_relation_outputs` 的逐行 Python 循环与 CUT-FGW；学生只占约 2%。现已将关系
抽取向量化为 `(node, event_id)` 稳定排序 + segment gather，降约 31 倍。

### 20.3 工程修复

- （v4 已删除）`--teacher-relation-warmup-epochs`：曾用于给旧的无关系头 checkpoint 热启动
  关系头；因教师现在始终带关系头，该路径已移除。
- 加载教师时若 CLI 的 `--teacher-hidden/--teacher-layers` 与 checkpoint 不一致会告警。
- `association` 缓冲区由 `torch.empty` 改为 `-1` 哨兵，消除未初始化值越界。

---

## 21. 学生架构扩展：轻量 1 层 GNN 学生

- `model/gnn_student.py: LightGNNStudent`：继承 LightST MLP，加一层邻居 mean-pooling
  （无 memory/attention/多层），0 初始化门控融合；含 `reset_state/update_state`。
- 训练/评测已支持有状态学生的 reset/update，无状态 MLP 路径不变。
- 实测（tgbl-wiki, 关系头教师）：GNN 学生 test MRR 0.5492，远高于图无关 MLP 0.38~0.40，
  但不再是图无关，且速度提升较小（含 1 跳邻居查询）。

---

## 22. 审查与修正：面向图无关 MLP 的蒸馏（v3）

### 22.1 三个组件的审查结论

**教师（`model/teacher.py`）——基本无问题。**
- `PyGTGNTeacher` = TGNMemory + 时间图注意力 + LinkPredictor + 关系投影头；
  固定 `relation_align_head` 使 `relation_projector` 在预训练阶段获得自监督梯度。
- 实验：重训带关系头后教师 val/test 由 0.7167/0.6735 升到 0.7249/0.6805。

**学生（`model/student.py`）——存在设计缺口。**
- 原图无关 MLP 只读 `src/candidates/timestamp`，没有任何结构输入或近期时序输入，
  这是它信息受限、蒸馏收益低的根因（val 0.4672 / test 0.3833）。

**蒸馏方法（`loss/`）——存在一个真实 bug。**
- 年龄锚点 bug：`anchor_relation_loss` 用原始年龄做最近邻；教师年龄按全局时间跨度
  归一化后近期历史都 ≈ 0，而学生槽年龄遍布 [0,1]，导致所有槽匹配到同一个教师 token，
  锚点退化为常数。修正为逐样本把两侧年龄各自重标定到 [0,1]。
- 定位澄清：关系 token 蒸馏（Anchor/CUT-FGW）适合结构化学生；对图无关 MLP 会引入噪声
  （强权重使 MLP test 从 0.3833 掉到 0.3473）。

### 22.2 离线结构/时序特征（`dataset/temporal.py: compute_structure_features`）

从训练集事件流统计每节点 5 维并标准化：出度、入度、`log1p(出+入)`、首次出现位置、
最近出现位置；训练集未出现的节点置 0。只依赖训练集（无泄漏），推理时作为静态输入，
仍图无关。通过 `--structure-features` 拼接到学生输入。

依据：GLNN（2110.08727）、InfGraND（2601.08033，一次性预计算多跳结构特征）、
L-STEP（2506.08309，可学习时空位置编码，TGB SOTA MLP）。

### 22.3 Anchor Relation 年龄归一化修正（`loss/kd.py`）

```
t' = (a_T - min_valid a_T) / (max_valid a_T - min_valid a_T)
s' = (a_S - min a_S) / (max a_S - min a_S)
distance = | s' - t' |,  j*(k) = argmin
```

### 22.4 MLP 蒸馏推荐配方

```
L = w_task·CE + w_logit·KL + w_rank·W1        （w_anchor = w_relation = 0）
student input = node_features ⊕ structure_features
```
脚本：`train-mlp-kd.sh`（蒸馏）、`train-mlp-struct.sh`（监督对照）；
结构化学生继续用 `train-anchor.sh`。

### 22.5 边界

- 结构特征为转导式（基于训练期统计）；新节点得到中性 0，属可控近似。
- 需在服务器上用 3 个 student seed 验证，并补 CPU / 大 batch 速度口径。

---

## 23. 代码清理（v4）

### 23.1 教师统一为带关系头

- `PyGTGNTeacher` 删除 `relation_align` 形参，`relation_align_head` **始终构建**，
  `model_config` 不再写 `relation_align` 字段。
- 删除 `train.py: warmup_teacher_relation`、`--teacher-relation-warmup-epochs` 参数及校验、
  `load_teacher_from_checkpoint(allow_missing_relation_align=...)` 兼容分支；加载教师改为
  始终 `strict=True`。
- 影响：`train_teacher_epoch` 的 `use_relations` 仅由 `relation_aux_weight > 0` 决定；
  旧的无关系头 checkpoint 不再兼容（需用带关系头的 `teacher.pt`）。
- 因为关系头**不进入打分路径**，`baseline` 教师的 logits/指标不受影响。

### 23.2 删除的死代码

- `util/metrics.py: ranking_metrics`（无调用方）。
- `util/__init__.py` 中 `count_trainable_parameters` / `count_all_parameters` /
  `count_runtime_state_elements` 的**再导出**（仅在 `model_parameter_summary` 内部使用，
  函数本身保留）。
- `loss/kd.py: rank_wasserstein_loss` 的 `top_weighted` 形参（无调用方传入 `False`），
  其加权逻辑内联保留。

### 23.3 保留

- `--method baseline`（KD-only 对照）、`LightGNNStudent`（`--student-arch gnn`）、
  `--structure-features` 与配套脚本/测试均保留。

