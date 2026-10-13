#!/usr/bin/env bash
# Variance screen A: no-code knobs on the supervised nbr2 student (full val).
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-stable.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-stable.sh
OUT="${OUT:-experiments/stable}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

BASE="--stage student --dataset tgbl-wiki --student-epochs $EPOCHS --patience $PATIENCE --structure-features --structure-feature-set nbr2"

# name | seed | extra
jobs=(
  "d03_s42|42|--dropout 0.3"
  "d03_s43|43|--dropout 0.3"
  "d03_s44|44|--dropout 0.3"
  "h64_s42|42|--student-hidden 64"
  "h64_s43|43|--student-hidden 64"
  "h64_s44|44|--student-hidden 64"
  "wd3_s42|42|--weight-decay 1.0e-3"
  "wd3_s43|43|--weight-decay 1.0e-3"
  "wd3_s44|44|--weight-decay 1.0e-3"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name seed extra <<< "$job"
  echo "=== RUN $name (shard $SHARD) ==="
  python train.py $BASE $extra --student-seed "$seed" --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Supervised student test" "$OUT"/*.log 2>/dev/null
