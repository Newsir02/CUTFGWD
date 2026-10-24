#!/usr/bin/env bash
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=experiments/content
mkdir -p "$OUT"
BASE="--dataset tgbl-wiki --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt --student-arch gnn --student-aggregation kernel --student-neighbors 8 --student-hidden 32 --student-blocks 2 --student-epochs 20 --patience 5 --dropout 0.3 --structure-features --structure-feature-set basic --student-neighbor-store exact --anchor-weight 0 --logit-weight 0.5 --rank-weight 0.5 --diversity-weight 0.01"
OT="--stage distill --method anchor_ot $BASE --relation-weight 0.5 --fused-alpha 0.2 --ot-fast"
KD="--stage distill --method baseline $BASE --relation-weight 0 --diversity-weight 0"

run() {
  name=$1; seed=$2; shift 2
  python -u train.py "$@" --student-seed "$seed" \
    --output-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
}

(
  run otk_content01_s42 42 $OT --relation-content-weight 0.1
  run otk_content01_s46 46 $OT --relation-content-weight 0.1
) &
(
  run otk_content05_s42 42 $OT --relation-content-weight 0.5
  run otk_content05_s46 46 $OT --relation-content-weight 0.5
) &
(
  run kd_content05_s42 42 $KD --relation-content-weight 0.5
  run kd_content05_s46 46 $KD --relation-content-weight 0.5
) &
wait
echo ALL_DONE
grep -H 'test: official_mrr' "$OUT"/*.log
