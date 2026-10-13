#!/usr/bin/env bash
# EMA decay screen on the supervised nbr2 student (variance reduction).
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-ema.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-ema.sh
OUT="${OUT:-experiments/ema}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

BASE="--stage student --dataset tgbl-wiki --student-epochs $EPOCHS --patience $PATIENCE --structure-features --structure-feature-set nbr2"

# name | seed | ema_decay
jobs=(
  "ema999_s42|42|0.999"
  "ema999_s43|43|0.999"
  "ema999_s44|44|0.999"
  "ema9999_s42|42|0.9999"
  "ema9999_s43|43|0.9999"
  "ema9999_s44|44|0.9999"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name seed decay <<< "$job"
  echo "=== RUN $name (shard $SHARD) ==="
  python train.py $BASE --ema-decay "$decay" --student-seed "$seed" --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Supervised student test" "$OUT"/*.log 2>/dev/null
