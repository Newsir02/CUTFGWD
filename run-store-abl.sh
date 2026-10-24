#!/usr/bin/env bash
cd /data/ljh/CUTFGWD || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Graph
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=experiments/trid1b
mkdir -p "$OUT"
COMMON="--dataset tgbl-wiki --student-arch gnn --student-hidden 32 --student-blocks 2 --student-epochs 20 --patience 5 --dropout 0.3 --structure-features --structure-feature-set basic --eval-max-events 3000 --student-seed 42 --student-neighbor-store pyg"

CUDA_VISIBLE_DEVICES=1 python -u train.py --stage student $COMMON \
  --student-neighbors 8 --student-aggregation mean \
  --output-dir "$OUT/sup_mean_pyg" > "$OUT/sup_mean_pyg.log" 2>&1 &
p0=$!

CUDA_VISIBLE_DEVICES=1 python -u train.py --stage distill \
  --teacher-checkpoint checkpoints/wiki-tgn/teacher.pt --method baseline \
  $COMMON --student-neighbors 8 --student-aggregation kernel \
  --logit-weight 0.5 --rank-weight 0.5 --relation-weight 0 \
  --kernel-weight 0 --diversity-weight 0 \
  --output-dir "$OUT/kd_kernel_pyg" > "$OUT/kd_kernel_pyg.log" 2>&1 &
p1=$!

wait $p0
wait $p1
echo DONE
grep -H 'test: official_mrr' "$OUT"/*.log
