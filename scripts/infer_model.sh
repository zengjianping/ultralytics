#!/bin/bash

MODEL="/home/ezgolf/ProjectGolf/Projects/ultralytics/runs/detect/models/golf_ball/hybrid_task_20241120/weights/best.pt"
SOURCE="/data/ModelTrainData/GolfBall/ezgolf_task_20241219/images/video_20241219174915416"

yolo predict \
    model="$MODEL" source="$SOURCE" \
    save=false show=false save_txt=true \
    conf=0.1 iou=0.1 imgsz=640

