#!/bin/bash

CONFIG_FILE="configs/yolo_detect_config.json"
MODEL_TYPE="yolo"
MODEL_PATH="datas/models/yolov10m.pt"
INPUT_PATH="datas/images/image03"

python tools/yolo_detect.py \
    --config_file "$CONFIG_FILE" \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --input_path "$INPUT_PATH"

