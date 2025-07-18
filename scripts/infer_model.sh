#!/bin/bash

MODEL="runs/detect/airport_ir_r1/weights/best.pt"
SOURCE="/data/Datasets/ModelTrainData/airport/nanning/d2_ir/job_20250522/images"

yolo predict \
    model="$MODEL" source="$SOURCE" \
    save=false show=true save_txt=false \
    conf=0.1 iou=0.5 imgsz=640,6144 line_width=2

