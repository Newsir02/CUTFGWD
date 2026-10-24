#!/usr/bin/env bash
# Round-2: full-val confirmation + candidate relational structure distillation.
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="${OUT:-experiments/trid2}"
TEACHER="${TEACHER:-checkpoints/wiki-tgn/teacher.pt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SHARD="${SHARD:-0}"
NSHARD="${NSHARD:-1}"
mkdir -p "$OUT"

COMMON="--dataset tgbl-wiki --student-arch gnn --student-hidden 32 --student-blocks 2 --student-epochs $EPOCHS --patience $PATIENCE --dropout 0.3 --structure-features --structure-feature-set basic --student-neighbors 8 --student-neighbor-store exact"
SUP="--stage student $COMMON"
DIST="--stage distill $COMMON --teacher-checkpoint $TEACHER --method baseline"
KD="--logit-weight 0.5 --rank-weight 0.5 --relation-weight 0 --kernel-weight 0 --diversity-weight 0"

# name | kind | seed | aggregation | extra
jobs=(
  "sup_mean_s42|sup|42|mean|"
  "sup_mean_s43|sup|43|mean|"
  "sup_mean_s44|sup|44|mean|"
  "kd_kernel_s42|dist|42|kernel|$KD"
  "kd_kernel_s43|dist|43|kernel|$KD"
  "kd_kernel_s44|dist|44|kernel|$KD"
  "kd_mean_s42|dist|42|mean|$KD"
  "trsd_kd_s42|dist|42|kernel|$KD --candidate-structure-weight 0.5"
  "trsd_kd_s43|dist|43|kernel|$KD --candidate-structure-weight 0.5"
  "trsd_kd_s44|dist|44|kernel|$KD --candidate-structure-weight 0.5"
  "trsd_w01_s42|dist|42|kernel|$KD --candidate-structure-weight 0.1"
  "trid_w025_s42|dist|42|kernel|--logit-weight 0.5 --rank-weight 0.5 --relation-weight 0 --kernel-weight 0.25 --diversity-weight 0"
)

idx=-1
for job in "${jobs[@]}"; do
  idx=$((idx + 1))
  if [ $((idx % NSHARD)) -ne "$SHARD" ]; then continue; fi
  IFS='|' read -r name kind seed agg extra <<< "$job"
  echo "=== RUN $name (seed=$seed agg=$agg shard $SHARD) ==="
  if [ "$kind" = "sup" ]; then
    python -u train.py $SUP --student-aggregation "$agg" --student-seed "$seed" $extra --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  else
    python -u train.py $DIST --student-aggregation "$agg" --student-seed "$seed" $extra --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
  fi
  echo "=== DONE $name exit=$? ==="
done
echo "=== SHARD $SHARD COMPLETE ==="
grep -H 'test: official_mrr' "$OUT"/*.log 2>/dev/null || true
