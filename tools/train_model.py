import os, sys, math, copy
import shutil, time, json
import argparse, cv2
import numpy as np

work_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, work_dir)

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '/data/ModelTrainData/GolfBall'
    parser.add_argument('--dataset_dir', type=str, default=f'{data_dir}/hybrid_task_20241219')
    parser.add_argument('--model', type=str, default='yolo11s')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=str, default='640')
    parser.add_argument('--rect_mode', type=int, default=0)
    args = parser.parse_args()
    return args

def train_model(dataset_dir, model, epochs, imgsz, batch, rect_mode):
    # Load a model
    #model = YOLO(f"{model}.yaml")  # build a new model from YAML
    model = YOLO(f"datas/models/{model}.pt")  # load a pretrained model (recommended for training)
    #model = YOLO(f"{model}.yaml").load(f"datas/models/{model}.pt")  # build from YAML and transfer weights

    # Train the model
    data_file = os.path.join(dataset_dir, 'data.yaml')
    imgsz = [int(x) for x in imgsz.split(',')]
    imgsz = imgsz * 2 if len(imgsz) == 1 else imgsz[0:2]
    results = model.train(data=data_file, epochs=epochs, imgsz=imgsz, batch=batch,
        rect=rect_mode>0, align_short=rect_mode>1)
    print('RESULTS:\n', results)

    return True

def main(args):
    train_model(args.dataset_dir, args.model, args.epochs, args.imgsz, args.batch,
                args.rect_mode)

if __name__ == '__main__':
    args = parse_args()
    main(args)

