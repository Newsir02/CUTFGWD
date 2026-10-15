#!/usr/bin/env bash
# Powered test of OT on the structured (1-layer GNN) student: sup vs full, dropout 0.3, 3 seeds.
#   SHARD=0 NSHARD=2 CUDA_VISIBLE_DEVICES=0 bash run-gnn.sh
#   SHARD=1 NSHARD=2 CUDA_VISIBLE_DEVICES=1 bash run-gnn.sh
OUT="${OUT:-experiments/gnn}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

SUP="--stage student --dataset tgbl-wiki --student-arch gnn --student-neighbors 8 --student-epochs $EPOCHS --patience $PATIENCE --dropout 0.3 --structure-features"
DIST="--stage distill --dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch gnn --student-neighbors 8 --student-epochs $EPOCHS --patience $PATIENCE --dropout 0.3 --anchor-weight 0.0 --ot-fast --fused-alpha 0.20 --logit-weight 0.5 --rank-weight 0.5 --relation-weight 1.0 --diversity-weight 0.01"

# name | kind | seed
jobs=(
  "gnn_sup_s42|student|42"
  "gnn_sup_s43|student|43"
  "gnn_sup_s44|student|44"
  "gnn_full_s42|distill|42"
  "gnn_full_s43|distill|43"
  "gnn_full_s44|distill|44"
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
