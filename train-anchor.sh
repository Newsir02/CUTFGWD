unset OMP_NUM_THREADS
export OMP_NUM_THREADS=4

python train.py \
  --stage distill \
  --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --output-dir checkpoints/wiki-anchor-ot \
  --method anchor_ot \
  --anchor-weight 0.5 \
  --logit-weight 0.1 \
  --rank-weight 0 \
  --relation-weight 1.0 \
  --diversity-weight 0 \
  --student-hidden 16 \
  --student-layers 2 \
  --teacher-layers 2 \
  --teacher-hidden 32 \
  "$@"
