#!/usr/bin/env bash
# Ablation ladder for the graph-free MLP student on tgbl-wiki.
# Run inside the repo with the training env already active.
#   bash run-ablation-mlp.sh
# Outputs per-run logs and checkpoints under $OUT (default experiments/abl_mlp).

OUT="${OUT:-experiments/abl_mlp}"
SEED="${SEED:-42}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
mkdir -p "$OUT"

DISTILL_COMMON="--dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch mlp --student-seed $SEED"

sup () {
  name=$1; shift
  echo "=== RUN $name ==="
  python train.py --stage student --dataset tgbl-wiki --student-seed "$SEED" \
    --output-dir "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
}

kd () {
  name=$1; shift
  echo "=== RUN $name ==="
  python train.py --stage distill $DISTILL_COMMON --output-dir "$OUT/$name" "$@" \
    > "$OUT/$name.log" 2>&1
  echo "=== DONE $name exit=$? ==="
}

# --- supervised baselines (no teacher) ---
sup sup_no_feat
sup sup_feat --structure-features

# --- distillation ladder: prediction KD, then add OT-related terms one by one ---
kd kd_lr      --logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.0 --relation-weight 0.0 --diversity-weight 0.0
kd kd_r       --logit-weight 0.0 --rank-weight 0.0 --anchor-weight 0.0 --relation-weight 1.0 --diversity-weight 0.0
kd kd_lr_r    --logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.0 --relation-weight 1.0 --diversity-weight 0.0
kd kd_lr_a    --logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.5 --relation-weight 0.0 --diversity-weight 0.0
kd kd_lr_ar   --logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.5 --relation-weight 1.0 --diversity-weight 0.0
kd kd_lr_ard  --logit-weight 0.5 --rank-weight 0.5 --anchor-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01

echo "================ SUMMARY ================"
for f in "$OUT"/*.log; do
  printf "%-14s " "$(basename "$f" .log)"
  grep -E "Student test|Supervised student test" "$f" | tail -1
done
