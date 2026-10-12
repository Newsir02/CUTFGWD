#!/usr/bin/env bash
# Neighbour-aggregation features: supervised MLP basic vs nbr vs rich_nbr.
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-nbr.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-nbr.sh
OUT="${OUT:-experiments/nbr}"
SEED="${SEED:-42}"
EVAL_MAX="${EVAL_MAX:-3000}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

BASE="--dataset tgbl-wiki --student-seed $SEED --student-epochs $EPOCHS --patience $PATIENCE --eval-max-events $EVAL_MAX --structure-features"

jobs=(
  "sup_basic|"
  "sup_nbr|--structure-feature-set nbr"
  "sup_rich_nbr|--structure-feature-set rich_nbr"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name extra <<< "$job"
  echo "=== RUN $name (shard $SHARD) ==="
  python train.py --stage student $BASE $extra --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Supervised student test" "$OUT"/*.log 2>/dev/null
