#!/usr/bin/env bash
# Extend GNN OT confirmation to n=6: seeds 45/46/47 for {sup, rel=0.5, fa=0.5}.
# Shard across GPUs:
#   SHARD=0..3 NSHARD=4 CUDA_VISIBLE_DEVICES=<0|1> bash run-gnn6.sh
OUT="${OUT:-experiments/gnn6}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

SUP="--stage student --dataset tgbl-wiki --student-arch gnn --student-neighbors 8 --student-epochs $EPOCHS --patience $PATIENCE --dropout 0.3 --structure-features --structure-feature-set basic"
DIST="--stage distill --dataset tgbl-wiki --teacher-checkpoint $TEACHER --method anchor_ot --student-arch gnn --student-neighbors 8 --student-epochs $EPOCHS --patience $PATIENCE --dropout 0.3 --anchor-weight 0.0 --structure-feature-set basic --ot-fast --logit-weight 0.5 --rank-weight 0.5 --diversity-weight 0.01"

# name | kind | seed | extra
jobs=(
  "sup_s45|student|45|"
  "rel05_s45|distill|45|--relation-weight 0.5 --fused-alpha 0.20"
  "fa05_s45|distill|45|--relation-weight 1.0 --fused-alpha 0.50"
  "sup_s46|student|46|"
  "rel05_s46|distill|46|--relation-weight 0.5 --fused-alpha 0.20"
  "fa05_s46|distill|46|--relation-weight 1.0 --fused-alpha 0.50"
  "sup_s47|student|47|"
  "rel05_s47|distill|47|--relation-weight 0.5 --fused-alpha 0.20"
  "fa05_s47|distill|47|--relation-weight 1.0 --fused-alpha 0.50"
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
    python train.py $DIST $extra --student-seed "$seed" --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  fi
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "Student test|Supervised student test" "$OUT"/*.log 2>/dev/null
