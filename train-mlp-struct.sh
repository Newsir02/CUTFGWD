unset OMP_NUM_THREADS
export OMP_NUM_THREADS=4

# 监督 MLP 学生（带离线结构/时序特征），作为蒸馏的公平对照。
python train.py \
  --stage student \
  --dataset tgbl-wiki \
  --structure-features \
  --output-dir checkpoints/wiki-mlp-struct \
  "$@"
