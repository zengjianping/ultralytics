#!/bin/bash

DATA_DIR="/data/ModelTrainData/GolfBall"
DATASET_DIR="${DATA_DIR}/hybrid_task_20241221"

python tools/train_model.py \
    --dataset_dir $DATASET_DIR \
    --epochs 200 --imgsz 640

