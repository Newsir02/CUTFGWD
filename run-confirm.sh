#!/usr/bin/env bash
# Confirmation batch: seed robustness (42/43/44) + fused_alpha x diversity.
# Launch one shard per GPU:
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-confirm.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-confirm.sh
# Env: OUT TEACHER EVAL_MAX EPOCHS PATIENCE SHARD NSHARD
OUT="${OUT:-experiments/confirm}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EVAL_MAX="${EVAL_MAX:-3000}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

BASE="--dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch mlp --student-epochs $EPOCHS --patience $PATIENCE --eval-max-events $EVAL_MAX --ot-fast --logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.0 --relation-weight 1.0 --diversity-weight 0.01"

# name | extra args
jobs=(
  "fa020_s43|--fused-alpha 0.20 --student-seed 43"
  "fa005_s42|--fused-alpha 0.05 --student-seed 42"
  "fa010_s42|--fused-alpha 0.10 --student-seed 42"
  "fa020_s44|--fused-alpha 0.20 --student-seed 44"
  "fa005_s43|--fused-alpha 0.05 --student-seed 43"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name extra <<< "$job"
  echo "=== RUN $name (shard $SHARD) ==="
  python train.py --stage distill $BASE $extra --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Student test" "$OUT"/*.log 2>/dev/null
