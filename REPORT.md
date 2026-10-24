# CUTFGWD 工作总结报告

> 本文件汇总本阶段的**动机、方法、实验结论、参考文献**，并给出**可复现实验清单（第 5 节）**。
> 细粒度实验记录见 [`RESULTS.md`](RESULTS.md)；方法设计见 [`IDEA.md`](IDEA.md)。

---

## 0. 摘要

在时序图基准（TGB `tgbl-wiki`）的动态链路预测上，我们研究**把 TGN 教师的知识蒸馏进轻量学生**。
核心创新是 **CUT-FGW**（Causal Unbalanced Temporal Fused Gromov-Wasserstein）——用**非平衡最优传输（OT）**对齐教师的关系轨迹与学生的关系槽。

本阶段通过大量受控实验得到三条结论：

1. **评测方差是首要障碍**：学生 seed 方差可达 ~0.025，盖过一切待测效应；`--dropout 0.3` 把它压到 **0.0014**（约 18×）。
2. **在"有功效"的协议下，唯一稳健的大效应是"结构"**：1 层 GNN 学生比图无关 MLP 高 **+0.027**。
3. **OT（及所有蒸馏变体）在图上没有可靠增益**：GNN 上 +0.006（不显著），图无关 MLP 上 ≈0 甚至显著为负。

---

## 1. 动机

### 1.1 问题
- 动态链路预测（TGB）上，**TGN** 类教师精度高，但推理依赖 **memory + 时间注意力 + 邻居采样**，延迟/内存大，难部署。
- 相反，**MLP** 无图依赖、推理快，但缺少**结构 + 时序**信息，精度低。
- 因此希望用**知识蒸馏（KD）**把教师的"结构与动态规律"转给学生，兼顾**精度**与**轻量/可部署**。

### 1.2 为什么用最优传输（OT）
- 教师的关系信息是**变长的历史 token 轨迹**，学生是**固定槽**，两者**元素数不等、无配对**。
- 逐元素 MSE/余弦无法对齐不等长集合；**OT（尤其 Gromov-Wasserstein 类）天然适合对齐"两个结构不同的点集"**，同时对齐**特征**与**关系结构**。
- 故提出 **CUT-FGW**：用**非平衡 Sinkhorn + Fused GW** 把教师关系轨迹的**二阶结构**（token 间关系 + 时间差分动态）迁移到学生固定槽。

### 1.3 要回答的核心问题
> 把 TGN 蒸馏进轻量学生是否有效？OT 是否是有效的蒸馏目标？在**图无关 MLP** 还是**结构化学生**上更有效？

---

## 2. 方法

### 2.1 教师：`PyGTGNTeacher`
- `TGNMemory`（IdentityMessage + LastAggregator）维护节点记忆 `memory`、`last_update`；
- `GraphAttentionEmbedding`（堆叠 `TransformerConv`）基于历史消息生成时间图嵌入；
- `LinkPredictor`：源/候选嵌入 → 候选 logits；
- `relation_projector`：把 `[neighbor_memory ‖ edge_attributes]` 投影为 `relation_dim` 的**关系 token**；
- `relation_align_head`（固定组件）：教师自监督对齐用，**不进入打分路径**。
- 训练目标：`L_CE + w_rel · L_align`（`L_align` 为关系表示与源嵌入的自监督对齐）。

### 2.2 学生
- **`LightSTMLPStudent`（图无关）**：只读 `src / candidates / timestamp`；
  `node_encoder(节点特征) + node_embeddings + TimeEncoding + 残差 MLP 块 + score_head`；
  关系分支：`relation_slots` 个固定槽（槽年龄由可学习 `age_gaps` 生成），`relation_decoder → relation_tokens`，`relation_mass_head → 槽质量`。
- **`LightGNNStudent`（结构化，消融/上界）**：在 MLP 上加**一层邻居 mean-pooling**（0 初始化门控），带状态（`reset_state/update_state`）。

### 2.3 输入特征（图无关、推理期静态，仅用训练集统计，无泄漏）
`--structure-feature-set`：
| 名称 | 内容 |
|---|---|
| `basic` | 出度、入度、活跃度(log1p)、首次/最近出现位置（5 维） |
| `rich` | basic + 唯一出/入邻居、二跳路径、互惠、PageRank、近期活跃度、位置 std（12 维） |
| `nbr / nbr2 / nbr3` | basic + **1/2/3 跳邻居特征均值**（SGC/GLNN 式预计算聚合） |
| `rich_nbr` | rich + 1 跳邻居均值 |

> 动机：让图无关学生以**静态向量**形式获得结构信息，推理时无需访问图/邻居。

### 2.4 蒸馏信号（`loss/distillation.py: CUTFGWDDistillation`）
```
L = w_task·L_sup(CE)
  + w_logit ·L_pred(KL)         # 预测蒸馏（≈GLNN 式）
  + w_rank  ·L_rank(W1)         # 候选排序关系
  + w_anchor·L_anchor            # 年龄锚定的关系表示对齐
  + w_relation·L_OT(CUT-FGW)     # 关系轨迹的结构最优传输
  + w_diversity·L_div            # 关系槽多样性正则
  + w_embed ·L_embed             # 特征级：学生投影表示 ↔ 教师上下文嵌入
```
权重为 0 的分支**真正跳过**（避免额外计算）。

### 2.5 CUT-FGW（`loss/cutfgw.py`）
- 教师侧：变长关系 token 轨迹（含年龄）；学生侧：固定有序槽；
- 代价 = `fused_alpha·属性代价(年龄)` + `(1-fused_alpha)·结构代价`；
  结构 = `memory_weight·(token 两两余弦关系矩阵)` + `dynamics_weight·(相邻 token 差分)`；
- 用**非平衡 Sinkhorn**（允许边际偏离，适配变长）在**FGW 近端迭代**中更新传输 plan；
- 关键修正：`anchor_relation_loss` 的年龄**逐样本重标定到 [0,1]**（否则所有槽塌缩到同一 token）。

### 2.6 评测协议与降方差
- **指标**：TGB **官方 MRR**（官方时间划分 + 官方负采样，协议不变）。
- **批量评测**：官方评测原为**逐事件**打分，改为**每批 16 事件**一起前向（`evaluator.eval` 仍逐事件），已验证与逐事件**数值一致**（差 4.5e-8），约 2× 提速。
- **val 子采样**（开发用）：`--eval-max-events` 在时间轴上**均匀抽样**；有状态模型（教师/GNN）自动改用连续前缀（满足事件索引连续要求）。
- **降方差**：`--dropout 0.3`（关键，见第 3 节）；学生 seed 配对复测。

---

## 3. 实验

**设置**：`tgbl-wiki`；教师 `checkpoints/wiki-tgn/teacher.pt`（TGN，test MRR **0.6806**），全程冻结；学生 20 epochs / patience 5；指标 = 官方 test MRR。

### 3.1 方差问题（表 13）
| 学生配置（监督，nbr2） | mean ± std（3 seed） |
|---|---|
| dropout 0.1（默认） | 0.4279 ± **0.0248** |
| **dropout 0.3** | 0.4423 ± **0.0014** |
| hidden 64 | 0.3906 ± 0.041 |
| weight-decay 1e-3 | 0.4163 ± 0.027 |

→ **`dropout 0.3` 同时提升均值并把方差降 18×**，是后续一切结论可信的前提。
（失败尝试：`dropout 0` 崩到 0.34–0.38；EMA 无改善；低 LR/强 WD 更差。）

### 3.2 有功效协议下的关键对比（dropout 0.3，3 seed，全量 val）
| 对比 | 效应 | 判定 |
|---|---|---|
| **结构：GNN vs MLP（sup）** | **+0.027**（0.4697 vs 0.4423） | **大且稳定** |
| MLP：特征 nbr2 vs basic | −0.003 | 无 |
| MLP nbr2：蒸馏 full vs sup | +0.002 | 不显著 |
| MLP basic：蒸馏 full vs sup | −0.009 | **显著为负** |
| GNN：蒸馏 full vs sup | +0.006 | 不显著 |

### 3.3 蒸馏消融（早期，dropout 0.1，seed 42，噪声较大）
- 结构特征曾"看似有效"（basic 0.4116 → nbr2 0.4538 +0.042）；单向 OT 曾"看似 +0.009"；
  但在 3.2 的**有功效协议下全部消失** → 之前是 seed 噪声。
- `anchor`（年龄锚定）**对 MLP 有害**；`diversity` 在噪声 regime 下看似有益。
- **`--ot-fast`**（fgw 4→2、sinkhorn 20→8）约 **4× 提速**，精度基本持平 → OT 计算是主要开销（约 +73ms/batch @fgw4/s20）。
- **特征级（嵌入）蒸馏**：直接在共享打分表示上以权重 1.0 对齐 → **崩溃 0.128**；改为**独立投影头 + 权重 0.1** → 0.4410（≈ sup，**无增益**）。

### 3.4 结论
1. **唯一稳健的大效应是"结构"**（GNN 比图无关 MLP 高 ~0.027）。
2. **邻域特征对图无关 MLP 无可靠增益**。
3. **对图无关 MLP，OT / 关系 token / anchor / diversity / 嵌入级蒸馏均无可靠增益**；对弱学生甚至有害。**但在 GNN 学生上调小 OT 权重后显著有效（见 3.5）。**
4. 图无关 MLP 天花板 ≈ **0.44–0.45**（GNN ≈ 0.47，教师 0.68）。
5. **方法学教训**：任何 ~0.01 量级的结论都必须**配对多 seed + 先验降方差**，否则会被 seed 噪声误导。

### 3.5 补充：GNN 上 OT 的 n=6 复核与跨数据集
- **tgbl-wiki，GNN 学生，n=6**：sup 0.4662 ± 0.0061；OT `rel=0.5` **0.4861 ± 0.0117（+0.0199，t≈5.5）**；`fa=0.5` 0.4916 ± 0.0236（+0.0254，t≈2.8）。此前的 +0.006 是因为 OT 权重过大（`relation_weight=1.0`）。
- **tgbl-subreddit 教师上限**：TGN 训到 50 epoch 仍只有 test 0.6215；TGN + 静态节点特征 val 0.683（未跑 test）；监督 GNN 教师 test **0.6615**（全量 val 复评 0.6639）。
- **3 seed（h32 / 8 邻居）**：sup **0.6534 ± 0.0010** vs OT rel=0.5 **0.6519 ± 0.0024**（Δ = −0.0015，ns）。改为**全量 val** 选择后 sup 升至 0.6583，OT 0.6578（Δ = −0.0005）。
- **余量控制**：把学生缩到 h16 / 4 邻居（余量 ~0.027），sup 0.6348 ± 0.0086 vs OT 0.6260 ± 0.0174（Δ = −0.0088，ns）；用"强 GNN logits + TGN 时序关系 token"的混合教师也无效（0.6256 ± 0.0113，Δ = −0.0092）。
- **结论**：subreddit 的教师 test 上限约 0.664，学生已达 0.658，**真实余量只有 ~0.006**；人为制造 ~0.03 余量后 OT 仍无收益。OT 的正向结果目前只在 wiki 成立；跨数据集复现的前提是"教师显著强于学生"，而 subreddit 不满足该前提。

### 3.6 方向调整（2026-10）：TGN → 轻量 GNN 的时序关系蒸馏

> 本节是新一轮研究（教师 TGN → 学生 1 层 GNN）的结论。完整方法/文献/实验见
> [`docs/TRKD.md`](docs/TRKD.md)，细粒度结果见 `RESULTS.md` 表 23–26。

**协议**：`tgbl-wiki`，full val 早停，官方 test MRR，配对 seed 42–47；教师
`checkpoints/wiki-tgn/teacher.pt`（test≈0.658）；学生 hidden 32 / 2 residual blocks /
K=8 / dropout 0.3 / 20 epochs；默认 exact 邻居存储。

| 方法 | mean ± std | 相对 sup |
|---|---|---|
| 监督（GNN，mean pooling） | 0.4520 ± 0.0199（n=6） | – |
| KD（logit+rank，kernel 聚合） | 0.4958 ± 0.0241（n=12） | **+0.043（n=6，t≈3.85）** |
| CUT-FGW OT + kernel 学生 | 0.4970 ± 0.0249（n=12） | **+0.051（n=6，t≈5.77）** |

- **OT + 时序注意力学生** 与 **KD** 都显著优于 sup；但 **OT+kernel 相对 KD 无差异**：
  n=6 +0.0139（t≈3.05），n=9 +0.0080（t≈1.49），
  **n=12 +0.0012（t≈0.21，6/12 胜出）**。新 seed 未复现，已撤回“OT 更优”的表述。
  OT 名义均值略高但计算显著更贵；**KD 是推荐方案**。
- 组合的机制假设：kernel 学生把邻居聚合权重放到**打分路径**；OT 迁移教师关系 token 的二阶结构；
  单独 kernel（0.475）与单独 KD 都不如组合，但协同效应在 n=12 下为 0，证据不足。
- **工程发现**：PyG `LastNeighborLoader` 在单 batch 度 > K 时会保留错误的“最近 K 条”；
  wiki 上 K=8 学生与教师支撑一致率仅 **77%**，修复后 **99.8%**。exact 支撑使 KD 在 s42
  从 0.5139 → 0.5247。
- **负结果（均有机制解释）**：
  - **TRID（教师邻居注意力质量）**：稳定为负/不显著。诊断显示注意力熵与教师 MRR
    **正相关（overall +0.368；成熟历史内 +0.151）**——注意力越集中，教师越容易错；
    不能把注意力质量当作关系知识。kernel-only 0.4706 < sup 0.4771。
  - **TRSD（候选嵌入两两结构）**：+0.005（ns）。
  - **TRDD（源表示 drift）**：n=6 配对 +0.0060（t≈0.50），s46 有 −0.052 的灾难 seed；
    随机 drift 对照 0.5180 < 真实 0.5409（s42），说明方向含信息但不稳定。
  - **PRKD（候选成对偏好）**：s42 有 +0.02，但 s43/s44 转负，不稳健。
  - **SERCD（共享支撑关系 token 内容，cos-only）**：s42 0.5434（+0.019）但 s46 0.4994（−0.005），
    不稳健；与 OT 同用反而更差（0.5175/0.4847）。
  - **绝对嵌入 KD**：0.4653（显著有害）；**支撑截断到 K=8/12**：未超过 K=24 KD。
- **结论**：在连续时间链路预测上，**候选级关系（logit/rank）与关系 token 结构（OT）**
  是可迁移的时序关系知识；**邻居注意力质量、候选结构、表示 drift** 不是可靠目标。
  OT+attention 是当前最优组合，但其 test 增益大于 val 增益（长时段泛化），仍需跨数据集确认。

---

## 4. 参考论文

**时序图 / 基准 / 教师**
- TGN — *Temporal Graph Networks for Deep Learning on Dynamic Graphs*，Rossi et al.，**arXiv:2006.10637**（教师结构来源）。
- TGB — *Temporal Graph Benchmark for Machine Learning on Temporal Graphs*，Huang et al.，**arXiv:2307.01026**（数据集/评测协议）。

**图→MLP 蒸馏（结构信息注入）**
- GLNN — *Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation*，Zhang et al.，**arXiv:2110.08727**（图无关 MLP + KD 的范式；对应 `L_pred`）。
- LLP — *Linkless Link Prediction via Relational Distillation*，Guo et al.，**arXiv:2210.05801**（链路预测的关系蒸馏；对应排序/关系匹配思想）。
- P&D — *Propagation-Embracing MLPs*（Propagate & Distill），Shin & Shin，**arXiv:2311.11759**（显式结构信息注入）。
- PGKD — *Edge-free but Structure-aware: Prototype-Guided KD*，Wu et al.，**arXiv:2303.13763**（无边的结构感知蒸馏）。
- InfGraND — *An Influence-Guided GNN-to-MLP Knowledge Distillation*，Eskandari et al.，**arXiv:2601.08033**（一次性**多跳邻域特征预计算**启发 `nbr*` 特征）。
- L-STEP — *Learnable Spatial-Temporal Positional Encoding for Link Prediction*，Tieu et al.，**arXiv:2506.08309**（TGB 上 MLP 的时空位置编码 SOTA，启发时序/结构特征）。

**最优传输 / 结构匹配**
- FGW（结构化 OT）— *Optimal Transport for structured data with application on graphs*，Vayer et al.，**arXiv:1805.09114**。
- FGW 理论 — *Fused Gromov-Wasserstein distance for structured objects*，Vayer et al.，**arXiv:1811.02834**。
- GW — *Gromov-Wasserstein averaging of kernel and distance matrices*，Peyré, Cuturi, Solomon，ICML 2016。
- Sinkhorn — *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*，Cuturi，NIPS 2013。
- 非平衡 OT — *Scaling Algorithms for Unbalanced Transport Problems*，Chizat et al.，**arXiv:1607.05816**（CUT-FGW 的非平衡 Sinkhorn 依据）。

**蒸馏 / 基础**
- KD — *Distilling the Knowledge in a Neural Network*，Hinton et al.，**arXiv:1503.02531**。
- FitNet — *FitNets: Hints for Thin Deep Nets*，Romero et al.，**arXiv:1412.6550**（提示/特征级蒸馏；对应 `L_embed`）。
- SGC — *Simplifying Graph Convolutional Networks*，Wu et al.，**arXiv:1902.07153**（一次预计算多跳传播；`nbr*` 的依据）。

**关系蒸馏 / 时序蒸馏（2026-10 新增）**
- RKD — *Relational Knowledge Distillation*，Park et al.，**arXiv:1904.05068**（距离/角度关系蒸馏；候选结构 TRSD 的依据）。
- Attention Transfer — *Paying More Attention to Attention*，Zagoruyko & Komodakis，**arXiv:1612.03928**（教师注意力匹配；TRID 的对照来源）。
- LightST — *Efficient Traffic Prediction Through Spatio-Temporal Distillation*，Zhang et al.，**arXiv:2501.10459**（时空 GNN→MLP KD；学生架构来源）。
- EasyST — *A Simple Framework for Spatio-Temporal Prediction*，Tang et al.，**arXiv:2409.06748**（时空 GNN→MLP KD）。
- LLP — 见上（候选级 rank/distribution 关系蒸馏；`L_rank` 的依据）。
- Transformer Hawkes Process — Zuo et al.，**arXiv:2002.09291**（时序影响核/条件强度的理论类比）。

---

## 5. 可复现实验清单（第 3 项：脚本 / 设置 / 结果）

### 5.1 环境
- 服务器：2×RTX 4090；conda env `Graph`（torch 2.1.0+cu118，torch_geometric 2.6.1，py-tgb 2.2.0）。
- 仓库根：`/data/ljh/CUTFGWD`；数据：`datasets/tgbl_wiki`；教师：`checkpoints/wiki-tgn/teacher.pt`。
- 本机到 GitHub 不通，代码在本地仓库编辑→push→`scp` 到服务器（仅在该目录内操作）。

### 5.2 关键默认（建议）
```bash
--dataset tgbl-wiki
--teacher-checkpoint checkpoints/wiki-tgn/teacher.pt
--method anchor_ot            # 默认开启结构特征
--student-arch mlp            # 图无关主线；gnn 作消融
--structure-feature-set nbr2  # 2 跳邻居均值（与 basic 无显著差异，可任选）
--dropout 0.3                 # 【关键】降方差
--student-epochs 20 --patience 5
--ot-fast                     # OT 4× 提速
--logit-weight 0.5 --rank-weight 0.5
--relation-weight 1.0 --diversity-weight 0.01 --anchor-weight 0.0
--embed-weight 0.0            # 若测特征级蒸馏：0.1
--eval-max-events 3000        # 仅开发期；正式报告用全量（去掉该参数）
# 配对多 seed：--student-seed 42/43/44 ...
```

### 5.3 实验脚本（仓库根）
| 脚本 | 用途 |
|---|---|
| `train-tgn.sh` | 训练带关系头的教师（`--method anchor_ot`） |
| `train-mlp-struct.sh` / `train-mlp-kd.sh` | 监督 MLP / MLP 蒸馏（含结构特征） |
| `train-anchor.sh` | 关系结构蒸馏（GNN/MLP） |
| `run-ablation-mlp.sh` | MLP 蒸馏消融（logit/rank/anchor/OT/diversity） |
| `run-ot-abl.sh` | OT 提速与 `fused_alpha` 扫描 |
| `run-confirm.sh` | 多 seed 确认 |
| `run-hop.sh` | 多跳邻居特征（nbr/nbr2/nbr3） |
| `run-kd-abl.sh` | nbr2 上的 KD 项消融 |
| `run-final.sh` | 全量 val + 3 seed 正式对比 |
| `run-stable.sh` | 降方差筛查（dropout/hidden/wd） |
| `run-d03.sh` / `run-basicd03.sh` | dropout 0.3 下的 OT 对比（nbr2 / basic） |
| `run-gnn.sh` | GNN 学生的功效化 OT 测试 |
| `run-embed.sh` | 特征级（嵌入）蒸馏 |
| `run-xsub-gnntea.sh` | subreddit 跨数据集复核（强监督 GNN 教师，sup vs rel=0.5，3 seed） |
| `run-xsub-fullval.sh` | subreddit 全量 val 复核对（单 seed 探针） |
| `run-xsub-small.sh` | subreddit 小学生在（h16 / 4 邻居，3 seed，余量控制） |
| `run-xsub-hybrid.sh` | subreddit 混合教师（GNN logits + TGN 关系 token） |

### 5.4 结果文件
- 所有实验输出在服务器 `experiments/<批次>/<配置>.log`（`Student test: official_mrr=...`）。
- 汇总表与结论：本仓库 `RESULTS.md`（表 1–17）。

### 5.5 评测/工程要点（已落地）
- 批量官方评测（`util/tgb_evaluation.py`，验证等价、~2× 提速）；有状态模型自动用连续前缀子采样。
- `--ot-fast`（OT 提速）；`--ema-decay`（EMA，实测无益，保留为可选）。
- 修复：`anchor_relation_loss` 年龄逐样本归一化；教师固定带关系头；nbr 特征广播 bug。

### 5.6 建议的下一步（供选题）
1. **论文叙事转向"结构 vs 图无关"**：这是唯一稳健的大效应；把 OT 作为"结构化学生"上的补充（需 n≥8 检验 +0.006）。
2. **重新设计蒸馏目标**：现有关系 token / 嵌入目标对图无关学生是噪声；可试 (a) 面向 MLP 的**师生表示对齐 + 强投影隔离**，或 (b) 面向 GNN 的**结构级蒸馏**并放大效果量。
3. 将 `--dropout 0.3` 设为全局默认，统一"配对多 seed + 全量 val"的评测协议。
