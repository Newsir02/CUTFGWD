#!/usr/bin/env bash
# Probe: does lowering LR / raising weight decay reduce seed variance? (supervised nbr2, full val)
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-lr.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-lr.sh
OUT="${OUT:-experiments/lr}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

BASE="--stage student --dataset tgbl-wiki --student-epochs $EPOCHS --patience $PATIENCE --structure-features --structure-feature-set nbr2"

# name | seed | lr | wd
jobs=(
  "lr3_s42|42|3.0e-4|1.0e-2"
  "lr3_s43|43|3.0e-4|1.0e-2"
  "lr3_s44|44|3.0e-4|1.0e-2"
  "lr1_s42|42|1.0e-4|1.0e-2"
  "lr1_s43|43|1.0e-4|1.0e-2"
  "lr1_s44|44|1.0e-4|1.0e-2"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name seed lr wd <<< "$job"
  echo "=== RUN $name (shard $SHARD) ==="
  python train.py $BASE --student-lr "$lr" --weight-decay "$wd" --student-seed "$seed" --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Supervised student test" "$OUT"/*.log 2>/dev/null
