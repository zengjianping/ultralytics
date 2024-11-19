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
    parser.add_argument('--dataset_dir', type=str, default=f'{data_dir}/ezgolf/task_video_20241115151137473')
    args = parser.parse_args()
    return args

def train_model(dataset_dir):
    # Load a model
    #model = YOLO("yolo11n.yaml")  # build a new model from YAML
    #model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Train the model
    data_file = os.path.join(dataset_dir, 'data.yaml')
    results = model.train(data=data_file, epochs=100, imgsz=640)
    print('RESULTS:\n', results)

    return True

def main(args):
    train_model(args.dataset_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)

