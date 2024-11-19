#!/bin/bash

DATA_DIR="/data/ModelTrainData/GolfBall"
DATASET_DIR="${DATA_DIR}/ezgolf/task_video_20241115151137473"

python tools/train_model.py \
    --dataset_dir $DATASET_DIR

