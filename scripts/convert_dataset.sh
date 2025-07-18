#!/bin/bash

WORK_MODE="merge"

DATA_DIR="/data/Datasets/ModelTrainData/airport"
DATASET_DIR="${DATA_DIR}/train_jobs/job_20250717"
SUB_DATASETS="\
    ${DATA_DIR}/nanning/d2_ir/job_20250522 \
    ${DATA_DIR}/nanning/d2_ir/job_20250603 \
"
CLASS_NAMES="aeroplane"

python tools/convert_dataset.py \
    --work_mode $WORK_MODE \
    --dataset_dir $DATASET_DIR \
    --sub_datasets $SUB_DATASETS \
    --class_names "$CLASS_NAMES"

