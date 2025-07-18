#!/bin/bash

DATA_DIR="/data/Datasets/ModelTrainData/airport"
DATASET_DIR="${DATA_DIR}/train_jobs/job_20250717"

python tools/train_model.py \
    --dataset_dir $DATASET_DIR \
    --model yolo11s --epochs 100 \
    --imgsz "640" --batch 4 --rect_mode 2

