#!/usr/bin/env bash
# OT-focused ablation: speed preset, diversity isolation, fused_alpha tweaks.
# Launch sharded across GPUs, e.g.:
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-ot-abl.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-ot-abl.sh
# Env: OUT SEED TEACHER EVAL_MAX EPOCHS PATIENCE SHARD NSHARD
OUT="${OUT:-experiments/ot_abl}"
SEED="${SEED:-42}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EVAL_MAX="${EVAL_MAX:-3000}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

BASE="--dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch mlp --student-seed $SEED --student-epochs $EPOCHS --patience $PATIENCE --eval-max-events $EVAL_MAX --logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.0 --relation-weight 1.0 --diversity-weight 0.0"

# name | extra args
jobs=(
  "ot_fast|--ot-fast"
  "ot_fast_div|--ot-fast --diversity-weight 0.01"
  "ot_fast_fa05|--ot-fast --fused-alpha 0.5"
  "ot_full_div|--diversity-weight 0.01"
  "ot_fast_fa005|--ot-fast --fused-alpha 0.05"
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
