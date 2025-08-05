import os, sys, math, copy
import glob, time, json
import argparse, cv2
import numpy as np
from easydict import EasyDict as edict
from ultralytics import YOLO, RTDETR, YOLOWorld, YOLOE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='yolo')
    parser.add_argument('--model_path', type=str, default='datas/models/yolo11s.pt')
    parser.add_argument('--config_file', type=str, default='configs/yolo_config_config.json')
    args = parser.parse_args()
    return args

def yolo_export(model_type, model_path, config_file):
    params = edict(json.loads(open(config_file).read()))

    if model_type == 'rtdetr':
        model = RTDETR(model_path)
    elif model_type == 'world':
        model = YOLOWorld(model_path)
    elif model_type == 'yoloe':
        model = YOLOE(model_path)
    else:
        model = YOLO(model_path)

    if params.limit_class:
        if model_type in ['world']:
            names = list(params.class_map.keys())
            model.set_classes(names)
        elif model_type in ['yoloe']:
            names = list(params.class_map.keys())
            model.set_classes(names, model.get_text_pe(names))
    model.info()

    imgsz = params.imgsz
    if isinstance(imgsz, str):
        imgsz = [int(s) for s in imgsz.split(',')]
    model.export(format=params.format, imgsz=params.imgsz, optimize=params.optimize,
        half=params.half, dynamic=params.dynamic, simplify=params.simplify, nms=params.nms,
        conf=params.conf, iou=params.iou, agnostic_nms=params.agnostic_nms, opset=params.opset)


def main(args):
    yolo_export(args.model_type, args.model_path, args.config_file)
    return True

if __name__ == '__main__':
    args = parse_args()
    main(args)

