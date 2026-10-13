#!/usr/bin/env bash
# Variance-reduction screen: dropout=0, full val, 3 seeds for sup and full recipes.
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-var.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-var.sh
OUT="${OUT:-experiments/var}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

SUP="--stage student --dataset tgbl-wiki --student-epochs $EPOCHS --patience $PATIENCE --dropout 0.0 --structure-features --structure-feature-set nbr2"
DIST="--stage distill --dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch mlp --student-epochs $EPOCHS --patience $PATIENCE --dropout 0.0 --anchor-weight 0.0 --structure-feature-set nbr2 --ot-fast --logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01 --fused-alpha 0.20"

# name | kind | seed
jobs=(
  "sup_d0_s42|student|42"
  "sup_d0_s43|student|43"
  "sup_d0_s44|student|44"
  "full_d0_s42|distill|42"
  "full_d0_s43|distill|43"
  "full_d0_s44|distill|44"
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
