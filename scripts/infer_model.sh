#!/bin/bash

MODEL="/data/ProjectGolf/Projects/ultralytics/runs/detect/models/golf_ball/ezgolf_video_20241115151137473/weights/best.pt"
SOURCE="/data/ModelTrainData/GolfBall/ezgolf/task_video_20241115151137473/images"

yolo predict \
    model="$MODEL" source="$SOURCE" \
    save=false show=false save_txt=true \
    conf=0.1 iou=0.1 imgsz=640

