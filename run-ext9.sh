#!/usr/bin/env bash
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=experiments/ext9
mkdir -p "$OUT"
COMMON="--dataset tgbl-wiki --student-arch gnn --student-aggregation kernel --student-neighbors 8 --student-hidden 32 --student-blocks 2 --student-epochs 20 --patience 5 --dropout 0.3 --structure-features --structure-feature-set basic --student-neighbor-store exact"
KD="--stage distill --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt --method baseline $COMMON --logit-weight 0.5 --rank-weight 0.5 --relation-weight 0 --kernel-weight 0 --diversity-weight 0"
OTK="--stage distill --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt --method anchor_ot $COMMON --anchor-weight 0 --logit-weight 0.5 --rank-weight 0.5 --relation-weight 0.5 --fused-alpha 0.20 --diversity-weight 0.01 --ot-fast"

(
  for seed in 48 49 50; do
    CUDA_VISIBLE_DEVICES=1 python -u train.py $KD --student-seed $seed \
      --output-dir "$OUT/kd_s${seed}" > "$OUT/kd_s${seed}.log" 2>&1
  done
) &
(
  for seed in 48 49 50; do
    CUDA_VISIBLE_DEVICES=1 python -u train.py $OTK --student-seed $seed \
      --output-dir "$OUT/otk_s${seed}" > "$OUT/otk_s${seed}.log" 2>&1
  done
) &
wait
echo ALL_DONE
grep -H 'test: official_mrr' "$OUT"/*.log
