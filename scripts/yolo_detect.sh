#!/bin/bash

#export PYTHONPATH=./:$PYTHONPATH

CONFIG_FILE="configs/yolo_detect_config.json"
MODEL_TYPE="ultralytics"
MODEL_PATH="datas/models/rtdetr-l.pt"
INPUT_PATH="datas/images/image01/team.jpg"

python tools/yolo_detect.py \
    --config_file "$CONFIG_FILE" \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --input_path "$INPUT_PATH"

