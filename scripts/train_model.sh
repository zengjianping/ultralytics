#!/bin/bash

DATA_DIR="/data/ModelTrainData/GolfBall"
DATASET_DIR="${DATA_DIR}/hybrid_task_20241219"

python tools/train_model.py \
    --dataset_dir $DATASET_DIR \
    --epochs 100 --imgsz 640

