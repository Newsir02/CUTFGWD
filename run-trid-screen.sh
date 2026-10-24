#!/usr/bin/env bash
# Round-1 screening on tgbl-wiki: uniform vs kernel aggregation, and
# temporal relational influence distillation (TRID) vs logit/OT baselines.
# Shard across processes with SHARD/NSHARD; all runs use the same teacher.
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="${OUT:-experiments/trid1}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-42}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

COMMON="--dataset tgbl-wiki --student-arch gnn --student-hidden 32 --student-blocks 2 --student-epochs $EPOCHS --patience $PATIENCE --dropout 0.3 --structure-features --structure-feature-set basic --eval-max-events 3000 --student-seed $SEED"
SUP="--stage student $COMMON"
DIST="--stage distill $COMMON --teacher-checkpoint $TEACHER --method baseline"

# name | kind | neighbors | aggregation | extra
jobs=(
  "sup_mean|sup|8|mean|"
  "sup_kernel|sup|8|kernel|"
  "kd_kernel|dist|8|kernel|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 0 --kernel-weight 0 --diversity-weight 0"
  "ot_mean|dist|8|mean|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 0.5 --fused-alpha 0.20 --diversity-weight 0.01 --kernel-weight 0 --ot-fast"
  "trid_k8|dist|8|kernel|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 0 --kernel-weight 1.0 --diversity-weight 0"
  "trid_k8_only|dist|8|kernel|--logit-weight 0 --rank-weight 0 --relation-weight 0 --kernel-weight 1.0 --diversity-weight 0"
  "trid_k24|dist|24|kernel|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 0 --kernel-weight 1.0 --diversity-weight 0"
  "sup_k24|sup|24|mean|"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name kind nbr agg extra <<< "$job"
  echo "=== RUN $name (nbr=$nbr agg=$agg shard $SHARD) ==="
  if [ "$kind" = "sup" ]; then
    python -u train.py $SUP --student-neighbors "$nbr" --student-aggregation "$agg" $extra --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  else
    python -u train.py $DIST --student-neighbors "$nbr" --student-aggregation "$agg" $extra --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  fi
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H -E "test" "$OUT"/*.log 2>/dev/null | grep -E "Student test|Supervised student test" || true
