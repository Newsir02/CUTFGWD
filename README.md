# CUTFGWD：时序图神经网络向 MLP / 小模型的蒸馏

本项目研究**时序图神经网络（Temporal GNN，以 TGN 为代表）到轻量模型的知识蒸馏**，
目标是在动态链路预测任务上，把 TGN 的预测能力与**关系知识**迁移到不依赖（或只依赖一跳）
图结构的轻量学生模型中，兼顾精度与推理效率。

核心方法：**Anchor Relation + Structural OT** 的统一蒸馏框架。

---

## 1. 背景与动机

- TGN 通过**记忆模块 + 时间图注意力**在连续时间事件流上取得很强的链路预测性能，
  但推理需要多跳历史邻居与记忆维护，延迟高、依赖图结构，难以部署。
- 图无关 MLP 推理快、无图依赖，但缺少结构/关系信息，精度差距大。
- 因此：把 TGN 蒸馏到 MLP / 轻量小模型，并**显式迁移关系表示与关系结构**。

---

## 2. 方法总览

### 2.1 教师与学生

| 角色 | 模型 | 说明 |
|---|---|---|
| 教师 | `PyGTGNTeacher` | PyG TGN：`TGNMemory` + `TransformerConv` 时间图注意力 + `LinkPredictor` + 关系投影头 `relation_projector` |
| 学生 A | `LightSTMLPStudent` | **图无关 MLP**：只读节点特征、节点 ID、查询时间；含固定关系槽 |
| 学生 B | `LightGNNStudent` | 在 MLP 学生上加**一层邻居 mean-pooling**（无 memory、无 attention、无多层消息传递） |

### 2.2 教师关系头（固定组件）

教师只用预测损失训练时，`relation_projector` 从未获得梯度（是随机的）。因此教师
**始终带关系对齐头** `relation_align_head`（不再有无关系头的变体）：

- 教师预训练时请求关系输出，并用**自监督对齐**训练关系投影/对齐头：
  `L_align = 1 - cos(A(pool(z_T)), sg(h_s)) + ||A(pool(z_T)) - sg(h_s)||²`，
  其中 `z_T` 为历史关系 token，`h_s` 为打分用的源节点嵌入，`sg` 为 stop-gradient。
- `--teacher-relation-weight` 控制该辅助项权重（`baseline` 默认 0，即只保留随机初始化
  的关系头、不训练它；`anchor_ot` 默认 1）。
- 关系头不参与教师打分路径，因此是否训练它**不改变教师预测**。

> 注意：旧的无关系头 checkpoint 已不再兼容，改用重新训练（带关系头）的 `teacher.pt`。

### 2.3 三条蒸馏信号

1. **Prediction Distillation**：对齐候选预测分布（KL）。
2. **Anchor Relation Distillation**：教师历史关系 token 与学生固定关系槽都按“年龄”排序，
   以**年龄为锚点**做最近邻匹配后对齐表示（cos + MSE），迁移关系**内容**。
3. **Structural OT Distillation**：复用 `CausalUnbalancedTemporalFGW`（CUT-FGW），
   对教师变长关系轨迹与学生固定槽的**二阶结构**（token 间关系矩阵 + 时间差分动态关系矩阵）
   做非平衡 Sinkhorn + FGW 最优传输，迁移关系**结构**。

另含排序关系项（Rank-Wasserstein）与关系槽多样性正则。

### 2.4 总损失

```
L = w_task · L_sup(CE)
  + w_logit · L_pred(KL)
  + w_rank  · L_rank(W1, 候选排序关系)
  + w_anchor· L_anchor(年龄锚定关系表示)
  + w_relation · L_OT(CUT-FGW 结构最优传输)
  + w_diversity · L_div(关系槽多样性)
```

损失实现见 `loss/distillation.py: CUTFGWDDistillation`；权重为 0 的分支会被**真正跳过**。

### 2.5 蒸馏流程

```
事件流 batch → 教师(冻结)打分 → 教师关系轨迹
                    │
             学生前向（MLP 或轻量 GNN）
                    │
   L_pred + L_rank + L_anchor + L_OT + L_div  → 反向传播
                    │
        官方验证集 MRR 早停 → 保存最优学生 → 官方测试集 MRR
```

---

## 3. 与普通知识蒸馏 / 原 CUT-FGW 的区别

| 维度 | 普通 KD | 原 CUT-FGW | 本方法 |
|---|---|---|---|
| 迁移对象 | 仅 logits | 关系结构（OT） | logits + 排序关系 + 关系表示 + 关系结构 |
| 教师关系头 | 不涉及 | 随机、未训练 | 自监督训练 / 热启动 |
| 关系对齐层次 | — | 仅二阶结构 | 一阶表示（Anchor）+ 二阶结构（OT） |
| 学生随机性 | 与教师共用种子 | 与教师共用种子 | 独立 `--student-seed` |
| 方法切换 | — | 固定 | `--method baseline\|anchor_ot`，默认路径不变 |

---

## 4. 代码目录

```
CUTFGWD/
├── train.py                     # 训练/蒸馏主入口（--stage / --method / --student-arch ...）
├── train-tgn.sh                 # 教师预训练脚本
├── train-mlp.sh                 # 监督 MLP 学生脚本
├── train-kd.sh                  # 原 CUT-FGW 蒸馏 baseline 脚本
├── train-anchor.sh              # 新方法 Anchor Relation + Structural OT 脚本
├── IDEA.md                      # 方法详细设计文档（含公式、消融、诊断）
├── README.md
├── dataset/
│   ├── temporal.py              # 事件流 Bundle、候选批加载器（训练负采样 / TGB）
│   ├── tgb_temporal.py          # TGB 数据集加载（PyG TemporalData）
│   └── synthetic_temporal.py    # 合成时序图（CPU smoke 测试）
├── model/
│   ├── teacher.py               # PyGTGNTeacher：TGN 教师 + 关系投影头（可选对齐头）
│   ├── student.py               # LightSTMLPStudent：图无关 MLP 学生
│   ├── gnn_student.py           # LightGNNStudent：轻量 1 层 GNN 学生
│   ├── tgn.py                   # GraphAttentionEmbedding + LinkPredictor
│   └── layers.py                # MLP / 残差块 / 时间编码
├── loss/
│   ├── distillation.py          # 总损失 CUTFGWDDistillation（零权重跳过）
│   ├── kd.py                    # KL / Rank-Wasserstein / 关系对齐 / Anchor / 多样性
│   └── cutfgw.py                # CausalUnbalancedTemporalFGW（结构 OT）
├── util/
│   ├── checkpoint.py            # 模型 checkpoint 存取与校验
│   ├── metrics.py               # MRR / Hits 指标
│   ├── tgb_evaluation.py        # TGB 官方负采样评测
│   ├── training.py              # 延迟/吞吐基准、参数统计
│   └── seed.py                  # 随机种子
├── tests/                       # 单元与流程测试
├── docs/                        # 设计/计划文档
└── artifacts/                   # 演示材料
```

---

## 5. 快速开始

### 5.1 环境

- Python 3.10、PyTorch、`torch-geometric`、`py-tgb`（评测 TGB 时需要）
- 数据：TGB `tgbl-wiki` 等，放在 `datasets/`（已加入 `.gitignore`）

### 5.2 Baseline（与原始实现一致）

```bash
# 教师
bash train-tgn.sh
# 监督 MLP 学生
python train.py --stage student --dataset tgbl-wiki --output-dir checkpoints/wiki-mlp
# 原 CUT-FGW 蒸馏
python train.py --stage distill --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --method baseline --output-dir checkpoints/wiki-kd-baseline
```

### 5.3 新方法（Anchor Relation + Structural OT）

```bash
# 训练带关系头的教师（教师始终带关系头）
bash train-tgn.sh --method anchor_ot

# 重用该教师做 Anchor Relation + Structural OT 蒸馏
python train.py --stage distill --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --method anchor_ot --teacher-relation-weight 1.0 \
  --anchor-weight 0.5 --logit-weight 0.3 --relation-weight 0.5 \
  --sinkhorn-iterations 10 --fgw-iterations 2 \
  --student-seed 42 --output-dir checkpoints/wiki-anchor-ot-s42

# 轻量 GNN 学生
python train.py --stage distill --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --method anchor_ot --student-arch gnn --student-neighbors 8 \
  --output-dir checkpoints/wiki-anchor-ot-gnn-s42
```

### 5.4 关键参数

| 参数 | 含义 |
|---|---|
| `--method {baseline,anchor_ot}` | 方法切换；`baseline` 与旧行为一致 |
| `--student-arch {mlp,gnn}` | 学生架构：图无关 MLP / 轻量 1 层 GNN |
| `--teacher-relation-weight` | 教师关系头自监督权重（`baseline` 默认 0，`anchor_ot` 默认 1） |
| `--anchor-weight` | Anchor Relation 蒸馏权重 |
| `--relation-weight` | Structural OT（CUT-FGW）权重 |
| `--logit-weight` / `--rank-weight` | 预测蒸馏 / 排序关系蒸馏权重 |
| `--student-seed` | 学生独立随机种子（配对实验用） |

---

## 6. 数据集与评测

- 数据集：Temporal Graph Benchmark（TGB），如 `tgbl-wiki`。
- 划分与负采样：**保持 TGB 官方时间划分与官方负采样协议不变**。
- 指标：官方 MRR（`util/tgb_evaluation.py`）。
- 公平性：`--seed` 固定数据与负采样，`--student-seed` 取 42/43/44 做配对比较。

---

## 7. 测试

```bash
python -m pytest tests -q
```

- 覆盖：TGN 教师、LightST MLP 学生、轻量 GNN 学生、Anchor 关系损失、
  CUT-FGW 反传、训练阶段流程、checkpoint 契约、数据契约等。
- 合成数据 smoke 可在 CPU 上完成 `data → teacher → student → relation → CUT-FGW → loss → backward → step`。

---

## 8. 更多

方法细节、公式、消融设计与实测诊断（参数量口径、蒸馏耗时分解、关系头修复）见
[`IDEA.md`](IDEA.md)。

---

## 9. 版本更新：MLP 蒸馏修正（v3）

### 9.1 审查结论

- **教师**：TGN + 关系头，工作正常。重训带关系头后教师略升
  （val/test：0.7167/0.6735 → **0.7249/0.6805**），关系头可作为辅助/正则。
- **学生**：原图无关 MLP **只读节点特征 + 节点 ID + 全局时间**，没有任何结构或
  近期时序输入——这是它"结构感知缺失、蒸馏难提升"的根因。
- **蒸馏方法**：发现并修复一个 **Anchor Relation 的年龄锚点 bug**。教师年龄按
  **全局**时间跨度归一化后，单条近期历史内的年龄都接近 0；而学生关系槽年龄遍布
  `[0,1]`。直接按原始年龄做最近邻匹配会让**所有学生槽都匹配到同一个（最近的）
  教师 token**，锚点匹配退化为常数。现已改为**逐样本把两侧年龄各自重标定到 [0,1]**。

### 9.2 主要修改

1. **离线结构/时序特征（`--structure-features`）**：从**训练集**事件流统计每节点
   5 维特征——出度、入度、活跃度（log1p）、首次出现位置、最近出现位置，标准化后
   拼接到图无关学生的输入。推理时无需访问图，即让 MLP 具备"结构感知 + 近期活跃度"。
   （参考 GLNN、InfGraND、L-STEP。）
2. **修复 `anchor_relation_loss` 的年龄归一化**（见 9.1）。
3. **明确 MLP 蒸馏配方**：图无关 MLP 使用"结构特征 + 预测/排序关系蒸馏"；
   关系 token / CUT-FGW 更适合**结构化学生**（1 层 GNN），对 MLP 会注入噪声。

### 9.3 用法

```bash
# 监督 MLP（带结构特征），作为公平对照
bash train-mlp-struct.sh

# MLP 蒸馏推荐配方（结构特征 + logit/rank KD，关闭关系 token/OT）
bash train-mlp-kd.sh

# 结构化学生（1 层 GNN）仍用关系结构蒸馏
python train.py --stage distill --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --method anchor_ot --student-arch gnn --student-neighbors 8 \
  --output-dir checkpoints/wiki-anchor-ot-gnn-s42
```

### 9.4 相关文献

- GLNN：Graph-less Neural Networks（arXiv:2110.08727）
- LLP：Linkless Link Prediction via Relational Distillation（arXiv:2210.05801）
- P&D：Propagation-Embracing MLPs（arXiv:2311.11759）
- PGKD：Edge-free but Structure-aware（arXiv:2303.13763）
- InfGraND：Influence-Guided GNN-to-MLP KD（arXiv:2601.08033）
- L-STEP：Learnable Spatial-Temporal Positional Encoding（arXiv:2506.08309）

---

## 10. 代码清理（v4）

- **教师统一为带关系头的变体**：删除 `PyGTGNTeacher(relation_align=...)` 开关，
  `relation_align_head` 始终构建；删除仅供“旧无关系头 checkpoint”使用的
  `--teacher-relation-warmup-epochs`、`warmup_teacher_relation` 与
  `load_teacher_from_checkpoint(allow_missing_relation_align=...)` 兼容分支。
- **删除死代码**：`util/metrics.py: ranking_metrics`；`util/__init__.py` 中未使用的
  `count_trainable_parameters` / `count_all_parameters` / `count_runtime_state_elements`
  再导出；`loss/kd.py: rank_wasserstein_loss` 恒不使用的 `top_weighted` 形参。
- 行为不变：关系头不参与教师打分路径，因此删除开关不影响 `baseline` 的教师预测；
  `--method baseline`、轻量 GNN 学生、结构特征均保留。
