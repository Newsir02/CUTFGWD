unset OMP_NUM_THREADS
export OMP_NUM_THREADS=4

python train.py \
  --stage distill \
  --dataset tgbl-wiki \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt \
  --output-dir checkpoints/wiki-KD \
  --student-hidden 16 \
  --student-layers 2 \
  --teacher-layers 2\
  --teacher-hidden 32 
