#!/usr/bin/env bash
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=experiments/pairwise
mkdir -p "$OUT"
BASE="--stage distill --dataset tgbl-wiki --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt --method baseline --student-arch gnn --student-aggregation kernel --student-neighbors 8 --student-hidden 32 --student-blocks 2 --student-epochs 20 --patience 5 --dropout 0.3 --structure-features --structure-feature-set basic --student-neighbor-store exact --relation-weight 0 --kernel-weight 0 --diversity-weight 0"

run() {
  name=$1; seed=$2; shift 2
  python -u train.py $BASE --student-seed "$seed" "$@" \
    --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
}

(
  run pr05_s42 42 --logit-weight 0.5 --rank-weight 0.5 --pairwise-weight 0.5
  run pr05_s46 46 --logit-weight 0.5 --rank-weight 0.5 --pairwise-weight 0.5
) &
(
  run pr05nr_s42 42 --logit-weight 0.5 --rank-weight 0 --pairwise-weight 0.5
  run pr05nr_s46 46 --logit-weight 0.5 --rank-weight 0 --pairwise-weight 0.5
) &
(
  run pr025_s42 42 --logit-weight 0.5 --rank-weight 0.5 --pairwise-weight 0.25
  run pr025_s46 46 --logit-weight 0.5 --rank-weight 0.5 --pairwise-weight 0.25
  run pr05t2_s42 42 --logit-weight 0.5 --rank-weight 0.5 --pairwise-weight 0.5 --pairwise-temperature 2.0
) &
wait
echo ALL_DONE
grep -H 'test: official_mrr' "$OUT"/*.log
