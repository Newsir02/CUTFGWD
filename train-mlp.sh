unset OMP_NUM_THREADS
export OMP_NUM_THREADS=4

python train.py \
  --stage student \
  --dataset tgbl-wiki \
  --output-dir checkpoints/wiki-mlp \
  --student-hidden 16 \
  --student-layers 2
