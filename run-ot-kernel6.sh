#!/usr/bin/env bash
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=experiments/otk
mkdir -p "$OUT"
BASE="--stage distill --dataset tgbl-wiki --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt --method anchor_ot --student-arch gnn --student-aggregation kernel --student-neighbors 8 --student-hidden 32 --student-blocks 2 --student-epochs 20 --patience 5 --dropout 0.3 --structure-features --structure-feature-set basic --student-neighbor-store exact --anchor-weight 0 --logit-weight 0.5 --rank-weight 0.5 --relation-weight 0.5 --fused-alpha 0.20 --diversity-weight 0.01 --ot-fast"

(
  for seed in 44 45; do
    CUDA_VISIBLE_DEVICES=1 python -u train.py $BASE --student-seed $seed \
      --output-dir "$OUT/ot_kernel_s${seed}" > "$OUT/ot_kernel_s${seed}.log" 2>&1
  done
) &
(
  for seed in 46 47; do
    CUDA_VISIBLE_DEVICES=1 python -u train.py $BASE --student-seed $seed \
      --output-dir "$OUT/ot_kernel_s${seed}" > "$OUT/ot_kernel_s${seed}.log" 2>&1
  done
) &
wait
echo ALL_DONE
grep -H 'test: official_mrr' "$OUT"/*.log
