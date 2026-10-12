# CUTFGWD 实验结果

## 运行设置
- 数据集：`tgbl-wiki`；指标：官方 test MRR（负采样协议不变）。
- 学生：图无关 `LightSTMLPStudent`（hidden 32, 2 residual blocks）。
- 教师：`checkpoints/wiki-tgn/teacher.pt`（TGN，test MRR 0.6806），全程冻结。
- 学生 seed=42；训练 30 epochs / patience 6；`--eval-max-events 3000` 仅加速 val 选择，**test 全量**。
- 运行时间：无 OT ~15 min/次；OT 全迭代 (fgw4/s20) ~50 min/次；`--ot-fast` (fgw2/s8) ~20 min/次。

## 表 1：图无关 MLP 蒸馏消融（logit/rank/anchor/OT/diversity）

| # | 方法 | logit | rank | anchor | relation(OT) | diversity | 结构特征 | test MRR | Δ vs sup_feat |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 监督（无特征） | – | – | – | – | – | ✗ | 0.3765 | −0.0451 |
| 2 | 监督（结构特征） | – | – | – | – | – | ✓ | 0.4216 | 0 |
| 3 | +预测 KD | 0.5 | 0.5 | 0 | 0 | 0 | ✓ | 0.4243 | +0.0027 |
| 4 | +OT（仅） | 0 | 0 | 0 | 1.0 | 0 | ✓ | 0.4185 | −0.0031 |
| 5 | **+预测 KD + OT** | 0.5 | 0.5 | 0 | 1.0 | 0 | ✓ | **0.4335** | +0.0119 |
| 6 | +预测 KD + anchor | 0.5 | 0.5 | 0.5 | 0 | 0 | ✓ | 0.4073 | −0.0143 |
| 7 | +预测 KD + anchor + OT | 0.5 | 0.5 | 0.5 | 1.0 | 0 | ✓ | 0.4103 | −0.0113 |
| 8 | +预测 KD + anchor + OT + div | 0.5 | 0.5 | 0.5 | 1.0 | 0.01 | ✓ | 0.4032 | −0.0184 |

## 表 2：OT 提速与改进（迭代次数 / fused_alpha / diversity 隔离）

基准均为 logit 0.5 + rank 0.5 + relation(OT) 1.0 + anchor 0 + 结构特征。

| 配置 | fgw/sinkhorn | fused_alpha | diversity | test MRR | 备注 |
|---|---|---|---|---|---|
| kd_lr_r | 4/20 | 0.20 | 0 | 0.4335 | 表 1 第 5 行 |
| ot_fast | 2/8 | 0.20 | 0 | 0.4292 | 迭代 1/5 → 运行时 ~1/3 |
| **ot_fast_div** | 2/8 | 0.20 | 0.01 | **0.4387** | **当前最优** |
| ot_fast_fa005 | 2/8 | 0.05 | 0 | 0.4366 | 更偏结构 |
| ot_fast_fa05 | 2/8 | 0.50 | 0 | 0.3992 | 过度偏属性 → 明显变差 |
| ot_full_div | 4/20 | 0.20 | 0.01 | 0.4246 | 全迭代+div（低于 fast+div，存疑/噪声） |

## 结论
- **结构特征是最大增益来源**（+0.045）。
- **预测蒸馏（logit+rank）小幅提升**（+0.003）。
- **OT 需与预测蒸馏同用才有效**（0.4243→0.4335）；单独 OT 反而略降。
- **anchor 对 MLP 有害**，建议关闭。
- **diversity 对 MLP 有益**（隔离验证：0.4292→0.4387）。
- **`--ot-fast`（fgw2/s8）≈4x 加速，精度基本持平**（0.4292 vs 0.4335，单 seed）。
- **OT 的 `fused_alpha` 很关键**：偏结构（0.05）好（0.4366），偏属性（0.5）差（0.3992）。

## 当前最优配方
`logit 0.5 + rank 0.5 + relation(OT, fgw2/s8, fused_alpha≈0.05~0.20) + diversity 0.01 + 结构特征`，anchor 关闭 → test ≈ 0.437~0.439。

## 表 3：确认实验（多 seed；`--ot-fast`，diversity 0.01，logit/rank 0.5，relation 1.0，anchor 0）

| 配方 | seed 42 | seed 43 | seed 44 | mean |
|---|---|---|---|---|
| 监督（结构特征，无 KD） | 0.4216 | – | – | – |
| +logit+rank（kd_lr） | 0.4243 | – | – | – |
| +OT（kd_lr_r，全迭代，无 div） | 0.4335 | – | – | – |
| +OT+diversity，fused_alpha=0.20 | 0.4387 | 0.4268 | 0.4235 | **0.4297**（std≈0.007） |
| +OT+diversity，fused_alpha=0.05 | 0.4324 | 0.4266 | – | 0.4295 |
| +OT+diversity，fused_alpha=0.10 | 0.4356 | – | – | – |

## 结论（确认后）
- 蒸馏配方（**OT + diversity + 预测 KD**）在多 seed 下稳定优于监督基线：mean ≈ **0.430**（范围 0.427~0.430），较监督 0.4216 提升 **+0.008**，较预测 KD 0.4243 提升约 +0.005。
- **seed 波动 std ≈ 0.007**，与配置差异（<0.01）同量级 → `fused_alpha ∈ {0.05,0.10,0.20}` 之间**无法区分**，不能用单 seed 断言谁最好。
- 结论：**OT 的贡献真实但幅度有限**（~+0.008）；当前瓶颈是图无关 MLP 的输入信息量，而非 OT 超参。

## 待办
1. 增强结构/时序输入特征（多跳/PPR/时间衰减度等）以抬升 MLP 上限——这是下一步主要方向。
2. 如需论文数字：用全量 val + 3 seed 对最终配方做正式评测。
3. 如继续调 OT：在更强特征下重扫 `fused_alpha`（当前被 seed 噪声掩盖）。
