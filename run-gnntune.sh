#!/usr/bin/env bash
# GNN student: tune OT strength / fused_alpha (seed 42, dropout 0.3, full val).
# Runs sequentially (single-GPU friendly): NSHARD=1.
#   CUDA_VISIBLE_DEVICES=1 bash run-gnntune.sh
OUT="${OUT:-experiments/gnntune}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
mkdir -p "$OUT"

BASE="--stage distill --dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch gnn --student-neighbors 8 --student-epochs $EPOCHS --patience $PATIENCE --dropout 0.3 --anchor-weight 0.0 --structure-feature-set basic --ot-fast --logit-weight 0.5 --rank-weight 0.5 --student-seed 42"

# name | extra
jobs=(
  "gw2|--relation-weight 2.0 --diversity-weight 0.01 --fused-alpha 0.20"
  "gw05|--relation-weight 0.5 --diversity-weight 0.01 --fused-alpha 0.20"
  "fa05|--relation-weight 1.0 --diversity-weight 0.01 --fused-alpha 0.50"
  "fa005|--relation-weight 1.0 --diversity-weight 0.01 --fused-alpha 0.05"
)

for job in "${jobs[@]}"; do
  IFS='|' read -r name extra <<< "$job"
  echo "=== RUN $name ==="
  python train.py $BASE $extra --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
done
echo "=== COMPLETE ==="
grep -H -E "Student test" "$OUT"/*.log 2>/dev/null
