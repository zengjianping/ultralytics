import os, sys, math, copy
import shutil, time, json
import argparse, cv2
import numpy as np

work_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, work_dir)

from ultralytics.data.utils import autosplit, check_det_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '/data/ModelTrainData/GolfBall'
    parser.add_argument('--work_mode', type=str, default='check')
    parser.add_argument('--dataset_dir', type=str, default=f'{data_dir}/ezgolf/task_video_20241115151137473')
    args = parser.parse_args()
    return args

def split_dataset(dataset_dir):
    image_dir = os.path.join(dataset_dir, 'images')
    autosplit(path=image_dir, weights=(0.9, 0.1, 0.0), annotated_only=True)

def check_dataset(dataset_dir):
    desc_file = os.path.join(dataset_dir, 'data.yaml')
    data = check_det_dataset(desc_file, autodownload=False)
    print(data)

def main(args):
    if args.work_mode == 'split':
        split_dataset(args.dataset_dir)
    elif args.work_mode == 'check':
        check_dataset(args.dataset_dir)
    return True

if __name__ == '__main__':
    args = parse_args()
    main(args)

