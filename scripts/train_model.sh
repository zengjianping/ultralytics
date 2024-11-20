#!/bin/bash

DATA_DIR="/data/ModelTrainData/GolfBall"
DATASET_DIR="${DATA_DIR}/ezgolf_task_20241117"
#DATASET_DIR="${DATA_DIR}/roboflow_golf_ball.v3i"
#DATASET_DIR="${DATA_DIR}/hybrid_task_20241120"

python tools/train_model.py \
    --dataset_dir $DATASET_DIR

