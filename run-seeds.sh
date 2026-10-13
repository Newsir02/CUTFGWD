#!/usr/bin/env bash
# Extend to n=6 seeds (45/46/47) for sup and full recipes (dropout default 0.1, full val).
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-seeds.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-seeds.sh
OUT="${OUT:-experiments/final}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

SUP="--stage student --dataset tgbl-wiki --student-epochs $EPOCHS --patience $PATIENCE --structure-features --structure-feature-set nbr2"
DIST="--stage distill --dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch mlp --student-epochs $EPOCHS --patience $PATIENCE --anchor-weight 0.0 --structure-feature-set nbr2 --ot-fast --logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01 --fused-alpha 0.20"

jobs=(
  "sup_s45|student|45"
  "sup_s46|student|46"
  "sup_s47|student|47"
  "full_s45|distill|45"
  "full_s46|distill|46"
  "full_s47|distill|47"
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
