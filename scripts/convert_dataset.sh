#!/bin/bash

WORK_MODE="merge"

DATA_DIR="/data/ModelTrainData/GolfBall"
DATASET_DIR="${DATA_DIR}/hybrid_task_20241221"
SUB_DATASETS="\
    ${DATA_DIR}/roboflow_golf_ball.v3i \
    ${DATA_DIR}/ezgolf_task_20241119 \
    ${DATA_DIR}/ezgolf_task_20241218 \
    ${DATA_DIR}/ezgolf_task_20241117 \
    ${DATA_DIR}/ezgolf_task_20241219 \
"

python tools/convert_dataset.py \
    --work_mode $WORK_MODE \
    --dataset_dir $DATASET_DIR \
    --sub_datasets $SUB_DATASETS

