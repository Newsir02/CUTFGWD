unset OMP_NUM_THREADS
export OMP_NUM_THREADS=4

# 图无关 MLP 学生的推荐蒸馏配方：
#   离线结构/时序特征 (InfGraND / L-STEP / GLNN) + 预测 & 排序关系蒸馏。
#   不启用针对"结构化学生"的关系 token / CUT-FGW，避免对 MLP 造成噪声。
python train.py \
  --stage distill \
  --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --method anchor_ot \
  --student-arch mlp \
  --structure-features \
  --logit-weight 0.5 \
  --rank-weight 0.5 \
  --anchor-weight 0.0 \
  --relation-weight 0.0 \
  --diversity-weight 0.0 \
  --student-seed 42 \
  --output-dir checkpoints/wiki-mlp-kd-s42 \
  "$@"
