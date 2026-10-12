#!/usr/bin/env bash
# Compare basic (5-dim) vs rich (12-dim) structure features on the supervised MLP,
# under the same (evenly-spaced) val subsample.
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-featset.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-featset.sh
OUT="${OUT:-experiments/featset}"
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
  "sup_rich|--structure-feature-set rich"
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
