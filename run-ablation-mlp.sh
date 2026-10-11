#!/usr/bin/env bash
# Sharded ablation ladder for the graph-free MLP student on tgbl-wiki.
# Meant to be launched in parallel across GPUs, e.g.:
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-ablation-mlp.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-ablation-mlp.sh
# Env knobs: OUT SEED TEACHER EVAL_MAX EPOCHS PATIENCE SHARD NSHARD
OUT="${OUT:-experiments/abl_mlp}"
SEED="${SEED:-42}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EVAL_MAX="${EVAL_MAX:-4000}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

# name | kind(student|distill) | extra args
jobs=(
  "sup_no_feat|student|"
  "sup_feat|student|--structure-features"
  "kd_lr|distill|--logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.0 --relation-weight 0.0 --diversity-weight 0.0"
  "kd_r|distill|--logit-weight 0.0 --rank-weight 0.0 --anchor-weight 0.0 --relation-weight 1.0 --diversity-weight 0.0"
  "kd_lr_r|distill|--logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.0 --relation-weight 1.0 --diversity-weight 0.0"
  "kd_lr_a|distill|--logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.5 --relation-weight 0.0 --diversity-weight 0.0"
  "kd_lr_ar|distill|--logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.5 --relation-weight 1.0 --diversity-weight 0.0"
  "kd_lr_ard|distill|--logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01"
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
    python train.py --stage distill --dataset tgbl-wiki \
      --teacher-checkpoint "$TEACHER" --method anchor_ot --student-arch mlp \
      --student-seed "$SEED" --student-epochs "$EPOCHS" --patience "$PATIENCE" \
      --eval-max-events "$EVAL_MAX" \
      --output-dir "$OUT/$name" $extra > "$OUT/$name.log" 2>&1
  fi
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Student test|Supervised student test" "$OUT"/*.log 2>/dev/null
