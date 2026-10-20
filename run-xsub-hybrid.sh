#!/usr/bin/env bash
# Hybrid-teacher probe on tgbl-subreddit:
#   logits  <- supervised GNN teacher (test 0.6615)
#   relations <- TGN teacher (temporal relation tokens)
# Student: GNN hidden16 / 2 blocks / 4 neighbors; seeds 123/2024 here (s42 in the probe).
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
mkdir -p experiments/xsub3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
run_one() {
  gpu=$1; seed=$2
  name=hybridH16_s${seed}
  echo "[$(date)] start $name on gpu$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -u train.py \
    --stage distill --dataset tgbl-subreddit \
    --student-arch gnn --student-hidden 16 --student-blocks 2 --student-neighbors 4 \
    --teacher-checkpoint checkpoints/sub-gnn-teacher/student_supervised.pt \
    --relation-teacher-checkpoint checkpoints/sub-tgn-long/teacher.pt \
    --method anchor_ot --student-epochs 10 --patience 4 \
    --dropout 0.3 --anchor-weight 0.0 --structure-feature-set basic --ot-fast \
    --logit-weight 0.5 --rank-weight 0.5 --diversity-weight 0.01 \
    --relation-weight 0.5 --fused-alpha 0.20 \
    --eval-max-events 3000 --student-seed $seed \
    --output-dir experiments/xsub3/$name > experiments/xsub3/$name.log 2>&1
  echo "[$(date)] $name exit=$? $(grep -E 'Student test' experiments/xsub3/$name.log | tail -1)"
}
run_one 0 123 &
p0=$!
run_one 1 2024 &
p1=$!
wait $p0
wait $p1
echo "[$(date)] HYBRID_SEEDS_DONE"