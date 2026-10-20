#!/usr/bin/env bash
# Full-val checkpoint selection probe (no --eval-max-events): matches the wiki protocol.
# sup vs OT rel=0.5, single seed 42, GNN hidden32 / 8 neighbors.
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
mkdir -p experiments/xsub4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
TEACHER=checkpoints/sub-gnn-teacher/student_supervised.pt

echo "[$(date)] start sup_s42 full-val"
CUDA_VISIBLE_DEVICES=0 python -u train.py --stage student --dataset tgbl-subreddit \
  --student-arch gnn --student-neighbors 8 --student-epochs 10 --patience 4 \
  --dropout 0.3 --structure-features --structure-feature-set basic \
  --student-seed 42 --output-dir experiments/xsub4/sup_s42 \
  > experiments/xsub4/sup_s42.log 2>&1 &
p0=$!
echo "[$(date)] start rel05_s42 full-val"
CUDA_VISIBLE_DEVICES=1 python -u train.py --stage distill --dataset tgbl-subreddit \
  --student-arch gnn --student-neighbors 8 --teacher-checkpoint $TEACHER \
  --method anchor_ot --student-epochs 10 --patience 4 \
  --dropout 0.3 --anchor-weight 0.0 --structure-feature-set basic --ot-fast \
  --logit-weight 0.5 --rank-weight 0.5 --diversity-weight 0.01 \
  --relation-weight 0.5 --fused-alpha 0.20 \
  --student-seed 42 --output-dir experiments/xsub4/rel05_s42 \
  > experiments/xsub4/rel05_s42.log 2>&1 &
p1=$!
wait $p0
echo "[$(date)] sup exit=$? $(grep -E 'Supervised student test' experiments/xsub4/sup_s42.log | tail -1)"
wait $p1
echo "[$(date)] rel05 exit=$? $(grep -E 'Student test' experiments/xsub4/rel05_s42.log | tail -1)"
echo "[$(date)] FULLVAL_PROBE_DONE"