#!/bin/bash

CONFIG_FILE="configs/yolo_export_config.json"
MODEL_TYPE="rtdetr"
MODEL_PATH="datas/models/rtdetr-l.pt"

python tools/yolo_export.py \
    --config_file "$CONFIG_FILE" \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH"

