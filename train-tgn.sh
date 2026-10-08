# #!/usr/bin/env bash
# set -euo pipefail
# cd "$(dirname "$0")"

# python train.py \
#   --stage "${STAGE:-all}" \
#   --seed "${SEED:-42}" \
#   --device "${DEVICE:-auto}" \
#   --dataset "${DATASET:-tgbl-wiki}" \
#   --data-root "${DATA_ROOT:-datasets}" \
#   --max-real-events "${MAX_REAL_EVENTS:-0}" \
#   --num-nodes "${NUM_NODES:-256}" \
#   --num-events "${NUM_EVENTS:-3000}" \
#   --feature-dim "${FEATURE_DIM:-16}" \
#   --edge-feature-dim "${EDGE_FEATURE_DIM:-16}" \
#   --num-candidates "${NUM_CANDIDATES:-16}" \
#   --history-size "${HISTORY_SIZE:-24}" \
#   --temporal-neighbors "${TEMPORAL_NEIGHBORS:-24}" \
#   --batch-size "${BATCH_SIZE:-128}" \
#   --num-workers "${NUM_WORKERS:-0}" \
#   --teacher-hidden "${TEACHER_HIDDEN:-64}" \
#   --student-hidden "${STUDENT_HIDDEN:-48}" \
#   --student-blocks "${STUDENT_BLOCKS:-3}" \
#   --time-dim "${TIME_DIM:-16}" \
#   --relation-dim "${RELATION_DIM:-64}" \
#   --relation-slots "${RELATION_SLOTS:-8}" \
#   --dropout "${DROPOUT:-0.10}" \
#   --teacher-epochs "${TEACHER_EPOCHS:-8}" \
#   --student-epochs "${STUDENT_EPOCHS:-15}" \
#   --teacher-lr "${TEACHER_LR:-1.0e-3}" \
#   --student-lr "${STUDENT_LR:-1.0e-3}" \
#   --weight-decay "${WEIGHT_DECAY:-1.0e-4}" \
#   --grad-clip "${GRAD_CLIP:-5.0}" \
#   --task-weight "${TASK_WEIGHT:-1.0}" \
#   --logit-weight "${LOGIT_WEIGHT:-1.0}" \
#   --rank-weight "${RANK_WEIGHT:-0.5}" \
#   --relation-weight "${RELATION_WEIGHT:-1.0}" \
#   --diversity-weight "${DIVERSITY_WEIGHT:-0.01}" \
#   --logit-temperature "${LOGIT_TEMPERATURE:-2.0}" \
#   --rank-temperature "${RANK_TEMPERATURE:-1.0}" \
#   --fused-alpha "${FUSED_ALPHA:-0.20}" \
#   --memory-weight "${MEMORY_WEIGHT:-1.0}" \
#   --dynamics-weight "${DYNAMICS_WEIGHT:-0.5}" \
#   --ot-entropy "${OT_ENTROPY:-0.08}" \
#   --marginal-relaxation "${MARGINAL_RELAXATION:-0.8}" \
#   --time-prior-scale "${TIME_PRIOR_SCALE:-0.20}" \
#   --fgw-iterations "${FGW_ITERATIONS:-4}" \
#   --sinkhorn-iterations "${SINKHORN_ITERATIONS:-20}" \
#   --teacher-checkpoint "${TEACHER_CHECKPOINT:-}" \
#   --output-dir "${OUTPUT_DIR:-checkpoints}" \
#   --latency-batches "${LATENCY_BATCHES:-20}" \
#   "$@"
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=4

python train.py \
  --stage teacher \
  --dataset tgbl-wiki \
  --output-dir checkpoints/wiki-tgn \
  --teacher-layers 2 \
  --teacher-hidden 32 \
  "$@"

