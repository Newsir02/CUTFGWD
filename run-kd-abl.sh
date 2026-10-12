#!/usr/bin/env bash
# KD-term ablation on nbr2 features (baseline comparison for the OT method).
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-kd-abl.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-kd-abl.sh
OUT="${OUT:-experiments/kd_abl}"
SEED="${SEED:-42}"
EVAL_MAX="${EVAL_MAX:-3000}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

BASE="--stage distill --dataset tgbl-wiki --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt --method anchor_ot --student-arch mlp --student-seed $SEED --student-epochs $EPOCHS --patience $PATIENCE --eval-max-events $EVAL_MAX --anchor-weight 0.0 --structure-feature-set nbr2 --ot-fast"

# name | extra weights
jobs=(
  "kd_logit|--logit-weight 1.0 --rank-weight 0.0 --relation-weight 0.0 --diversity-weight 0.0"
  "kd_rank|--logit-weight 0.0 --rank-weight 1.0 --relation-weight 0.0 --diversity-weight 0.0"
  "kd_logit_rank|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 0.0 --diversity-weight 0.0"
  "kd_lr_rel|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.0 --fused-alpha 0.20"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name extra <<< "$job"
  echo "=== RUN $name (shard $SHARD) ==="
  python train.py $BASE $extra --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Student test" "$OUT"/*.log 2>/dev/null
