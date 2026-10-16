#!/usr/bin/env bash
# Feature-level (embedding) distillation test: sup(baseline) vs emb vs emb+OT.
# nbr2 features, dropout 0.3, 3 seeds, full val.
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-embed.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-embed.sh
OUT="${OUT:-experiments/embed}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

DIST="--stage distill --dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch mlp --student-epochs $EPOCHS --patience $PATIENCE --dropout 0.3 --anchor-weight 0.0 --structure-feature-set nbr2 --ot-fast --fused-alpha 0.20"

# name | seed | weights
jobs=(
  "emb_s42|42|--embed-weight 1.0 --logit-weight 0.5 --rank-weight 0.5 --relation-weight 0.0 --diversity-weight 0.0"
  "emb_s43|43|--embed-weight 1.0 --logit-weight 0.5 --rank-weight 0.5 --relation-weight 0.0 --diversity-weight 0.0"
  "emb_s44|44|--embed-weight 1.0 --logit-weight 0.5 --rank-weight 0.5 --relation-weight 0.0 --diversity-weight 0.0"
  "embot_s42|42|--embed-weight 1.0 --logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01"
  "embot_s43|43|--embed-weight 1.0 --logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01"
  "embot_s44|44|--embed-weight 1.0 --logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name seed extra <<< "$job"
  echo "=== RUN $name (shard $SHARD) ==="
  python train.py $DIST $extra --student-seed "$seed" --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Student test" "$OUT"/*.log 2>/dev/null
