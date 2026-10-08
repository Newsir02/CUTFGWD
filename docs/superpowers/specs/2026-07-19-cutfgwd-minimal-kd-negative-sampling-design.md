# CUTFGWD 最小修改实验设计

日期：2026-07-19

## 1. 背景与问题

现有 `tgbl-wiki`、`seed=42` 结果为：

- 监督 LightST 学生：最佳验证 MRR `0.4549`，官方测试 MRR `0.3906`。
- 原始蒸馏学生：最佳验证 MRR `0.4586`，官方测试 MRR `0.3845`。
- 教师：官方测试 MRR `0.6836`。

蒸馏学生的验证结果并未明显弱于监督学生，但测试 MRR 低 `0.0061`。当前证据不足以证明蒸馏稳定有效。现有训练还存在两个可由最小修改直接隔离的干扰：

1. 原始联合目标使用 `logit_weight=1.0`、`rank_weight=0.5`、`relation_weight=1.0`，而训练日志中 KL 原始值约为 task loss 的 3.4 倍，教师信号可能压过真实标签。
2. 训练候选加载器每次迭代都用同一个 seed 重新初始化生成器，导致每个 epoch 重复相同负样本，限制了训练覆盖范围。
3. 教师预训练时不请求 relation 输出，因此教师 `relation_projector` 没有被教师任务训练；当前第一轮最小实验不使用该关系信号，以免把关系分支问题和 KL 权重问题混在一起。

本设计只验证“弱 KL + 动态训练负样本”能否让蒸馏学生稳定优于对应监督学生，不修复或重构 CUT-FGW。

## 2. 目标与成功标准

### 2.1 目标

- 保持教师、学生网络结构和现有教师 checkpoint 不变。
- 训练负样本随 epoch 可复现地变化。
- 将蒸馏目标缩减为真实标签 CE 加弱 KL。
- 用配对的三个学生随机种子比较 KD 与监督训练。
- 区分动态负样本带来的收益和知识蒸馏带来的额外收益。

### 2.2 成功标准

最小修改仅在同时满足以下条件时判定有效：

1. 最佳 KD 配置三个 student seed 的平均官方 test MRR 严格大于旧监督基线 `0.3906`。
2. 三个 seed 上 `KD test MRR - 对应 supervised test MRR` 的平均值严格大于 0。

单个 seed 的提升、验证集提升或只超过旧基线，均不足以判定有效。

## 3. 范围

### 3.1 允许修改

- `dataset/temporal.py`：训练候选的 epoch-aware 负采样。
- `loss/distillation.py`：零权重损失真正跳过计算。
- `train.py`：独立 student seed、训练 epoch 传递、按损失配置请求 relation 输出。
- `train-kd.sh`：设置弱 KL 默认值并允许命令行覆盖。
- `train-mlp.sh`：允许命令行覆盖。
- `tests/`：增加对应的单元与流程测试。

### 3.2 明确不修改

- TGN 教师和 LightST 学生网络结构。
- CUT-FGW、Rank-Wasserstein 和 Sinkhorn 的算法实现。
- 现有教师 checkpoint 及 checkpoint 格式。
- 官方 TGB 验证/测试负样本。
- 早停逻辑、官方 MRR 计算与数据切分。
- `logit_temperature=2.0`。

## 4. 训练负采样设计

### 4.1 CandidateBatchLoader 状态

在 `CandidateBatchLoader` 中增加：

- `self.epoch = 0`
- `set_epoch(epoch: int)`

`epoch` 必须为非负整数；非法值直接报错。

### 4.2 有效采样 seed

- 训练集：`effective_seed = base_seed + epoch`
- 验证集：始终使用构造时的固定 `base_seed`
- 测试集：始终使用构造时的固定 `base_seed`

因此：

- 同一 split、同一 epoch、同一 base seed 重放完全一致。
- 不同训练 epoch 生成不同候选。
- 验证和测试候选不随训练 epoch 改变。
- 监督学生与 KD 学生在相同 epoch 使用相同训练候选。

### 4.3 训练循环

在每个训练 epoch 开始前执行：

```python
train_loader.set_epoch(epoch - 1)
```

该规则应用于：

- 教师预训练循环（未来重新训练教师时保持一致）；
- 蒸馏学生循环；
- 监督学生循环。

本次正式 `tgbl-wiki` 最小实验复用现有教师 checkpoint，不重新训练教师。

## 5. 学生随机种子设计

新增参数：

```text
--student-seed
```

语义如下：

- `--seed`：数据构造、训练负采样基础 seed 和数据指纹 seed；正式实验固定为 42。
- `--student-seed`：学生参数初始化、学生 Dropout 和学生训练随机性；正式实验取 42、43、44。

若未传 `--student-seed`，默认回退到 `--seed`，保持旧命令兼容。

在构建学生模型之前调用 `seed_everything(student_seed)`。`student_seed` 不进入数据签名，因此现有 `seed=42` 教师 checkpoint 仍可加载；它会保存到学生 checkpoint 的 args 元数据中。

## 6. 零权重分支与关系输出

### 6.1 蒸馏目标

`CUTFGWDDistillation.forward` 始终计算 task CE。其余分支只有在对应权重非零时才计算：

- `logit_weight == 0`：不计算 KL，不访问教师 logits。
- `rank_weight == 0`：不计算 Rank-Wasserstein。
- `relation_weight == 0`：不计算 CUT-FGW，也不访问 relation tensors。
- `diversity_weight == 0`：不计算关系槽多样性。

被跳过的分量返回与学生 logits 同设备、同 dtype 的标量零，保留统一日志字段：`total`、`task`、`logit`、`rank`、`relation`、`diversity`。

### 6.2 是否请求 relation 输出

- 教师：仅当 `relation_weight != 0` 时请求 relation 输出。
- 学生：仅当 `relation_weight != 0` 或 `diversity_weight != 0` 时请求 relation 输出。

第一轮实验中 `relation_weight=0` 且 `diversity_weight=0`，因此教师和学生均使用 `return_relations=False`，不执行 relation projector、关系槽解码或 Sinkhorn。

## 7. 损失配置

seed=42 第一轮筛选固定：

| 实验 | task | logit | rank | relation | diversity |
|---|---:|---:|---:|---:|---:|
| 动态负样本监督组 | 1.0 | 0 | 0 | 0 | 0 |
| 弱 KL-0.1 | 1.0 | 0.1 | 0 | 0 | 0 |
| 弱 KL-0.25 | 1.0 | 0.25 | 0 | 0 | 0 |

Rank 和 CUT-FGW 本轮均关闭。该选择不是否定两种算法，而是先建立一个可解释、低干扰的 KD 基线；关系蒸馏修复属于后续独立实验。

`train-kd.sh` 默认显式传入：

```bash
--task-weight 1.0
--logit-weight 0.1
--rank-weight 0
--relation-weight 0
--diversity-weight 0
```

脚本末尾增加 `"$@"`，允许覆盖 `--logit-weight`、`--student-seed` 和 `--output-dir`。`train-mlp.sh` 同样增加 `"$@"`。

## 8. 实验矩阵与选择规则

### 8.1 seed=42 筛选

运行三组：

1. 动态负样本监督组。
2. 动态负样本、KL 权重 0.1。
3. 动态负样本、KL 权重 0.25。

仅根据官方验证 MRR 在 `0.1` 与 `0.25` 中选择最佳 KL 权重；不得使用测试集选择权重。若验证 MRR 相同，优先选择较小权重 `0.1`。

### 8.2 三 seed 正式比较

固定：

```text
--seed 42
```

分别运行：

```text
--student-seed 42
--student-seed 43
--student-seed 44
```

每个 student seed 都运行：

- 动态负样本监督组；
- 已选定 KL 权重的 KD 组。

报告：

- 每个 seed 的最佳验证 epoch、验证 MRR、官方 test MRR 和训练耗时；
- KD 与监督组各自 test MRR 的均值和样本标准差；
- 每个 seed 的 `KD - supervised` test MRR；
- 三个配对差值的平均值；
- 是否同时满足两个成功标准。

## 9. 测试设计

新增或扩展自动化测试以覆盖：

1. 同一训练 epoch 重放时负样本完全一致。
2. 不同训练 epoch 的负样本发生变化。
3. 验证集和测试集不随 `set_epoch` 改变。
4. `set_epoch` 拒绝负数和非整数。
5. 所有辅助权重为零时，只提供学生 logits 也能计算 CE 并反向传播。
6. `relation_weight=0` 时 CUT-FGW 不会被调用。
7. 相同 `student_seed` 的学生初始化一致，不同 seed 初始化不同。
8. 完整 synthetic quick teacher→distill 流程能够运行。
9. 运行全部现有测试，确保旧训练阶段和 checkpoint 行为不回归。

## 10. 验收顺序

必须按以下顺序执行：

1. 运行新增单元测试。
2. 运行全部现有测试。
3. 运行 synthetic quick teacher→distill smoke。
4. 在 `tgbl-wiki` 上用 student seed 42 运行三组筛选实验。
5. 按验证 MRR 选择最佳 KL 权重。
6. 最佳 KD 与监督组分别运行 student seed 42、43、44。
7. 汇总配对结果并应用成功标准。

当前 Windows 工作区没有 `datasets` 目录；正式 `tgbl-wiki` 三 seed 实验需要在拥有现有数据和 CUDA 环境的 Ubuntu 项目副本中执行。Windows 工作区负责代码修改、单元测试和 synthetic smoke；正式实验日志必须来自同一数据签名 `228a3c93cf46fab568950e0f64796b09338b2e467a992ab7689bfc4bbcb583a1`。

## 11. 示例命令

```bash
# 动态负样本监督组
bash train-mlp.sh \
  --seed 42 \
  --student-seed 42 \
  --output-dir checkpoints/wiki-mlp-dynamic-s42

# KL=0.1
bash train-kd.sh \
  --seed 42 \
  --student-seed 42 \
  --logit-weight 0.1 \
  --output-dir checkpoints/wiki-kd01-s42

# KL=0.25
bash train-kd.sh \
  --seed 42 \
  --student-seed 42 \
  --logit-weight 0.25 \
  --output-dir checkpoints/wiki-kd025-s42
```

每次运行使用独立输出目录，禁止覆盖旧基线 checkpoint。

## 12. 失败处理与结论边界

- 若自动化测试或 synthetic smoke 失败，停止正式实验并先修复回归。
- 若两个 KL 权重均未改善验证 MRR，仍选择验证较高者完成三 seed 统计，但最终结论按成功标准判定。
- 若平均 KD test MRR 未超过 `0.3906`，结论为“最小修改未超过旧基线”。
- 若平均 KD 超过 `0.3906` 但平均配对差值不为正，结论为“收益主要来自动态负样本，不能归因于蒸馏”。
- 若两个条件均满足，结论为“弱 KL + 动态负样本的最小修改有效”。
- 本实验不用于判断 CUT-FGW 或 Rank-Wasserstein 本身是否有效；两者需要后续单独修复和消融。
