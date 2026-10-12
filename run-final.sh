#!/usr/bin/env bash
# Noise-reduced final comparison on nbr2 features: FULL val + 3 seeds.
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-final.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-final.sh
# Env: OUT TEACHER EPOCHS PATIENCE SHARD NSHARD
OUT="${OUT:-experiments/final}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

SUP="--stage student --dataset tgbl-wiki --student-epochs $EPOCHS --patience $PATIENCE --structure-features --structure-feature-set nbr2"
DIST="--stage distill --dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch mlp --student-epochs $EPOCHS --patience $PATIENCE --anchor-weight 0.0 --structure-feature-set nbr2 --ot-fast"

# name | kind(student|distill) | seed | extra
jobs=(
  "sup_s42|student|42|"
  "sup_s43|student|43|"
  "sup_s44|student|44|"
  "lr_s42|distill|42|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 0.0 --diversity-weight 0.0"
  "lr_s43|distill|43|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 0.0 --diversity-weight 0.0"
  "lr_s44|distill|44|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 0.0 --diversity-weight 0.0"
  "rel_s42|distill|42|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.0 --fused-alpha 0.20"
  "rel_s43|distill|43|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.0 --fused-alpha 0.20"
  "rel_s44|distill|44|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.0 --fused-alpha 0.20"
  "full_s42|distill|42|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01 --fused-alpha 0.20"
  "full_s43|distill|43|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01 --fused-alpha 0.20"
  "full_s44|distill|44|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01 --fused-alpha 0.20"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name kind seed extra <<< "$job"
  echo "=== RUN $name (shard $SHARD) ==="
  if [ "$kind" = "student" ]; then
    python train.py $SUP --student-seed "$seed" --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  else
    python train.py $DIST --student-seed "$seed" $extra --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  fi
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Student test|Supervised student test" "$OUT"/*.log 2>/dev/null
