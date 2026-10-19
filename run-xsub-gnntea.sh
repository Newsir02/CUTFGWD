#!/usr/bin/env bash
# Cross-dataset (tgbl-subreddit) re-check with a strong supervised GNN teacher.
# Teacher: checkpoints/sub-gnn-teacher/student_supervised.pt
#   train with: --stage student --student-arch gnn --student-hidden 128
#   --student-blocks 2 --student-neighbors 16 --student-epochs 24 --patience 6
#   --dropout 0.2 --structure-features --structure-feature-set basic
# GPU0: sup seeds 42/123/2024; GPU1: OT rel=0.5 seeds 42/123/2024.
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
mkdir -p experiments/xsub2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
TEACHER=checkpoints/sub-gnn-teacher/student_supervised.pt

run_one() {
  gpu=$1; name=$2; shift 2
  echo "[$(date)] start $name on gpu$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -u train.py "$@" --output-dir experiments/xsub2/$name > experiments/xsub2/$name.log 2>&1
  echo "[$(date)] $name exit=$? $(grep -E 'Student test|Supervised student test' experiments/xsub2/$name.log | tail -1)"
}

gpu0_queue() {
  for seed in 42 123 2024; do
    run_one 0 sup_s${seed} --stage student --dataset tgbl-subreddit \
      --student-arch gnn --student-neighbors 8 --student-epochs 10 --patience 4 \
      --eval-max-events 3000 --dropout 0.3 --structure-features --structure-feature-set basic \
      --student-seed $seed
  done
}

gpu1_queue() {
  for seed in 42 123 2024; do
    name=rel05_s${seed}
    run_one 1 $name --stage distill --dataset tgbl-subreddit \
      --student-arch gnn --student-neighbors 8 --teacher-checkpoint $TEACHER \
      --method anchor_ot --student-epochs 10 --patience 4 \
      --eval-max-events 3000 --dropout 0.3 --anchor-weight 0.0 \
      --structure-feature-set basic --ot-fast --logit-weight 0.5 --rank-weight 0.5 \
      --diversity-weight 0.01 --relation-weight 0.5 --fused-alpha 0.20 \
      --student-seed $seed
  done
}

gpu0_queue &
p0=$!
gpu1_queue &
p1=$!
wait $p0
wait $p1
echo "[$(date)] ALL_DONE"