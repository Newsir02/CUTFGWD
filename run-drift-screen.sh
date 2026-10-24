#!/usr/bin/env bash
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=experiments/trid3
mkdir -p "$OUT"
COMMON="--stage distill --dataset tgbl-wiki --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt --method baseline --student-arch gnn --student-aggregation kernel --student-neighbors 8 --student-hidden 32 --student-blocks 2 --student-epochs 20 --patience 5 --dropout 0.3 --structure-features --structure-feature-set basic --student-seed 42 --student-neighbor-store exact --logit-weight 0.5 --rank-weight 0.5 --relation-weight 0 --kernel-weight 0 --diversity-weight 0"

CUDA_VISIBLE_DEVICES=1 python -u train.py $COMMON --drift-weight 0.1 \
  --output-dir "$OUT/drift_w01_s42" > "$OUT/drift_w01_s42.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python -u train.py $COMMON --drift-weight 0.25 \
  --output-dir "$OUT/drift_w025_s42" > "$OUT/drift_w025_s42.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=1 python -u train.py $COMMON --embed-weight 0.1 \
  --output-dir "$OUT/embed_w01_s42" > "$OUT/embed_w01_s42.log" 2>&1 &
p2=$!
wait $p0
wait $p1
wait $p2
echo DONE
grep -H 'test: official_mrr' "$OUT"/*.log
