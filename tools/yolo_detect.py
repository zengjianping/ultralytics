import os, sys, math, copy
import glob, time, json
import argparse, cv2
import numpy as np
from easydict import EasyDict as edict
from ultralytics import YOLO, RTDETR, YOLOWorld, YOLOE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='yolo')
    parser.add_argument('--model_path', type=str, default='datas/models/yolo11n.pt')
    parser.add_argument('--input_path', type=str, default='datas/images/cars.jpg')
    parser.add_argument('--config_file', type=str, default='configs/yolo_detect_config.json')
    args = parser.parse_args()
    return args

def yolo_detect(input_path, model_type, model_path, config_file):
    params = edict(json.loads(open(config_file).read()))
    if os.path.isdir(args.input_path):
        input_dir = args.input_path
    else:
        input_dir = os.path.dirname(input_path)
    input_name = 'test_results'
    class_ids = None

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
        else:
            class_ids = list(params.class_map.values())

    model.info()

    results = model(input_path, stream=False, project=input_dir, name=input_name,
        classes=class_ids, imgsz=params.image_size, conf=params.thres_conf,
        iou=params.thres_iou, max_det=params.max_det, save=params.save_image,
        save_txt=params.save_txt, save_conf=params.save_conf, show=params.show_image,
        show_labels=params.show_label, show_conf=params.show_conf)
    result_dir = results[0].save_dir
    
    print(f'Results saved in directory: {result_dir}')

    return results


def main(args):
    yolo_detect(args.input_path, args.model_type, args.model_path, args.config_file)
    return True

if __name__ == '__main__':
    args = parse_args()
    main(args)

