#!/bin/bash

#export PYTHONPATH=./:$PYTHONPATH

CONFIG_FILE="configs/yolo_detect_config.json"
MODEL_TYPE="ultralytics"
MODEL_PATH="datas/models/yolov10m.pt"
INPUT_PATH="datas/images/image00/small-vehicles1.jpeg"
INPUT_PATH="/home/zengjianping/ProjectZKZS/Projects/sahi/datas/test_images/airport_nanning_d02_vis_01_unfold/"

python tools/yolo_detect.py \
    --config_file "$CONFIG_FILE" \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --input_path "$INPUT_PATH"

