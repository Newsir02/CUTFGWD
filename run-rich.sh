#!/usr/bin/env bash
# Rich structure/temporal features: supervised vs distilled (vs basic-feature refs).
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-rich.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-rich.sh
# Env: OUT SEED TEACHER EVAL_MAX EPOCHS PATIENCE SHARD NSHARD
OUT="${OUT:-experiments/rich}"
SEED="${SEED:-42}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EVAL_MAX="${EVAL_MAX:-3000}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

# name | kind(student|distill) | extra
jobs=(
  "sup_rich|student|--structure-features"
  "ot_rich|distill|--ot-fast --fused-alpha 0.20 --diversity-weight 0.01"
  "ot_rich_fa005|distill|--ot-fast --fused-alpha 0.05 --diversity-weight 0.01"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name kind extra <<< "$job"
  echo "=== RUN $name (shard $SHARD) ==="
  if [ "$kind" = "student" ]; then
    python train.py --stage student --dataset tgbl-wiki --student-seed "$SEED" \
      --student-epochs "$EPOCHS" --patience "$PATIENCE" --eval-max-events "$EVAL_MAX" \
      --output-dir "$OUT/$name" $extra > "$OUT/$name.log" 2>&1
  else
    python train.py --stage distill --dataset tgbl-wiki --teacher-checkpoint "$TEACHER" \
      --method anchor_ot --student-arch mlp --student-seed "$SEED" \
      --student-epochs "$EPOCHS" --patience "$PATIENCE" --eval-max-events "$EVAL_MAX" \
      --logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.0 --relation-weight 1.0 \
      --output-dir "$OUT/$name" $extra > "$OUT/$name.log" 2>&1
  fi
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Student test|Supervised student test" "$OUT"/*.log 2>/dev/null
