#!/usr/bin/env bash
# Headroom-controlled subreddit test: hidden16 / 4-neighbor student, strong GNN teacher.
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
mkdir -p experiments/xsub3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
TEACHER=checkpoints/sub-gnn-teacher/student_supervised.pt

run_one() {
  gpu=$1; name=$2; shift 2
  echo "[$(date)] start $name on gpu$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -u train.py "$@" --output-dir experiments/xsub3/$name > experiments/xsub3/$name.log 2>&1
  echo "[$(date)] $name exit=$? $(grep -E 'Student test|Supervised student test' experiments/xsub3/$name.log | tail -1)"
}

gpu0_queue() {
  for seed in 42 123 2024; do
    run_one 0 supH16_s${seed} --stage student --dataset tgbl-subreddit \
      --student-arch gnn --student-hidden 16 --student-blocks 2 --student-neighbors 4 \
      --student-epochs 10 --patience 4 --dropout 0.3 \
      --structure-features --structure-feature-set basic \
      --eval-max-events 3000 --student-seed $seed
  done
}

gpu1_queue() {
  for seed in 42 123 2024; do
    name=rel05H16_s${seed}
    run_one 1 $name --stage distill --dataset tgbl-subreddit \
      --student-arch gnn --student-hidden 16 --student-blocks 2 --student-neighbors 4 \
      --teacher-checkpoint $TEACHER --method anchor_ot \
      --student-epochs 10 --patience 4 --dropout 0.3 --anchor-weight 0.0 \
      --structure-feature-set basic --ot-fast --logit-weight 0.5 --rank-weight 0.5 \
      --diversity-weight 0.01 --relation-weight 0.5 --fused-alpha 0.20 \
      --eval-max-events 3000 --student-seed $seed
  done
}

gpu0_queue &
p0=$!
gpu1_queue &
p1=$!
wait $p0
wait $p1
echo "[$(date)] SMALL_ALL_DONE"