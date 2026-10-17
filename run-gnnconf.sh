#!/usr/bin/env bash
# Confirm GNN OT configs on seeds 43/44 (sequential, single-GPU friendly).
#   CUDA_VISIBLE_DEVICES=1 bash run-gnnconf.sh
OUT="${OUT:-experiments/gnnconf}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
mkdir -p "$OUT"

BASE="--stage distill --dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch gnn --student-neighbors 8 --student-epochs $EPOCHS --patience $PATIENCE --dropout 0.3 --anchor-weight 0.0 --structure-feature-set basic --ot-fast --logit-weight 0.5 --rank-weight 0.5 --diversity-weight 0.01"

jobs=(
  "gw05_s43|43|--relation-weight 0.5 --fused-alpha 0.20"
  "gw05_s44|44|--relation-weight 0.5 --fused-alpha 0.20"
  "fa05_s43|43|--relation-weight 1.0 --fused-alpha 0.50"
  "fa05_s44|44|--relation-weight 1.0 --fused-alpha 0.50"
)

for job in "${jobs[@]}"; do
  IFS='|' read -r name seed extra <<< "$job"
  echo "=== RUN $name ==="
  python train.py $BASE $extra --student-seed "$seed" --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
done
echo "=== COMPLETE ==="
grep -H -E "Student test" "$OUT"/*.log 2>/dev/null
