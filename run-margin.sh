#!/usr/bin/env bash
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=experiments/margin
mkdir -p "$OUT"
BASE="--stage distill --dataset tgbl-wiki --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt --method baseline --student-arch gnn --student-aggregation kernel --student-neighbors 8 --student-hidden 32 --student-blocks 2 --student-epochs 20 --patience 5 --dropout 0.3 --structure-features --structure-feature-set basic --student-neighbor-store exact --relation-weight 0 --kernel-weight 0 --diversity-weight 0"

run() {
  name=$1; seed=$2; shift 2
  python -u train.py $BASE --student-seed "$seed" "$@" \
    --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
}

(
  for seed in 42 46 52; do
    run m05_s${seed} $seed --logit-weight 0.5 --rank-weight 0.5 --margin-weight 0.5
  done
) &
(
  for seed in 42 46 52; do
    run m10_s${seed} $seed --logit-weight 0.5 --rank-weight 0.5 --margin-weight 1.0
  done
) &
(
  for seed in 42 46 52; do
    run mnr_s${seed} $seed --logit-weight 0.5 --rank-weight 0 --margin-weight 0.5
  done
) &
wait
echo ALL_DONE
grep -H 'test: official_mrr' "$OUT"/*.log
