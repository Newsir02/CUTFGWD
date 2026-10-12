#!/usr/bin/env bash
# Multi-hop neighbour aggregation: supervised MLP nbr vs nbr2 vs nbr3.
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-hop.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-hop.sh
OUT="${OUT:-experiments/hop}"
SEED="${SEED:-42}"
EVAL_MAX="${EVAL_MAX:-3000}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

BASE="--stage student --dataset tgbl-wiki --student-seed $SEED --student-epochs $EPOCHS --patience $PATIENCE --eval-max-events $EVAL_MAX --structure-features"

jobs=(
  "sup_nbr|--structure-feature-set nbr"
  "sup_nbr2|--structure-feature-set nbr2"
  "sup_nbr3|--structure-feature-set nbr3"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name extra <<< "$job"
  echo "=== RUN $name (shard $SHARD) ==="
  python train.py $BASE $extra --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Supervised student test" "$OUT"/*.log 2>/dev/null
