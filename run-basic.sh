#!/usr/bin/env bash
# Regime where OT matters most: BASIC features (5-dim), sup vs full, seeds 43-45 (s42 exists).
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-basic.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-basic.sh
OUT="${OUT:-experiments/basic}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

SUP="--stage student --dataset tgbl-wiki --student-epochs $EPOCHS --patience $PATIENCE --structure-features --structure-feature-set basic"
DIST="--stage distill --dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch mlp --student-epochs $EPOCHS --patience $PATIENCE --anchor-weight 0.0 --structure-feature-set basic --ot-fast --logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01 --fused-alpha 0.20"

jobs=(
  "basic_sup_s43|student|43"
  "basic_sup_s44|student|44"
  "basic_sup_s45|student|45"
  "basic_full_s43|distill|43"
  "basic_full_s44|distill|44"
  "basic_full_s45|distill|45"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name kind seed <<< "$job"
  echo "=== RUN $name (shard $SHARD) ==="
  if [ "$kind" = "student" ]; then
    python train.py $SUP --student-seed "$seed" --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  else
    python train.py $DIST --student-seed "$seed" --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  fi
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Student test|Supervised student test" "$OUT"/*.log 2>/dev/null
