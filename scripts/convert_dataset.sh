#!/bin/bash

WORK_MODE="check"

DATA_DIR="/data/ModelTrainData/GolfBall"
DATASET_DIR="${DATA_DIR}/ezgolf/task_video_20241115151137473"

python tools/convert_dataset.py \
    --work_mode $WORK_MODE \
    --dataset_dir $DATASET_DIR

