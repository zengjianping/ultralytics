import os, sys, math, copy
import shutil, time, json
import argparse, cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

work_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, work_dir)

from ultralytics.data.utils import autosplit, check_det_dataset
from ultralytics.data.converter import convert_coco
from ultralytics.utils import TQDM


def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '/data/ModelTrainData/GolfBall'
    parser.add_argument('--work_mode', type=str, default='check')
    parser.add_argument('--dataset_dir', type=str, default=f'{data_dir}/ezgolf/task_video_20241115151137473')
    args = parser.parse_args()
    return args

def split_dataset(dataset_dir):
    image_dir = os.path.join(dataset_dir, 'images')
    autosplit(path=image_dir, weights=(0.9, 0.1, 0.0), annotated_only=False)

def check_dataset(dataset_dir):
    desc_file = os.path.join(dataset_dir, 'data.yaml')
    data = check_det_dataset(desc_file, autodownload=False)
    print(data)

def convert_coco_to_yolo(dataset_dir):
    # Create dataset directory
    annot_dir = os.path.join(dataset_dir, 'annotations')
    save_dir = os.path.join(annot_dir, 'coco_converted')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    default_json_file = os.path.join(annot_dir, 'instances_default.json')
    default_json_file = Path(default_json_file)

    lname = default_json_file.stem.replace("instances_", "")
    fn = save_dir / lname / "labels"  # folder name
    fn.mkdir(parents=True, exist_ok=True)
    with open(default_json_file) as f:
        data = json.load(f)

    # Create image dict
    images = {f'{x["id"]:d}': x for x in data["images"]}

    # Create image-annotations dict
    imgToAnns = {x["id"]: list() for x in data["images"]}
    for ann in data["annotations"]:
        imgToAnns[ann["image_id"]].append(ann)

    # Write labels file
    for img_id, anns in TQDM(imgToAnns.items(), desc=f"Annotations {default_json_file}"):
        img = images[f"{img_id:d}"]
        h, w = img["height"], img["width"]
        file_name:str = img["file_name"]
        npos = file_name.find('images/')
        if npos >= 0:
            file_name = file_name[npos+7:]

        bboxes = []
        for ann in anns:
            if ann.get("iscrowd", False):
                continue
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(ann["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            cls = ann["category_id"] - 1  # class
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)

        # Write
        txt_file:Path = (fn / file_name).with_suffix(".txt")
        txt_file.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_file, "a") as file:
            for i in range(len(bboxes)):
                line = (*(bboxes[i]),)  # cls, box or segments
                file.write(("%g " * len(line)).rstrip() % line + "\n")


def convert_yolo_to_coco(dataset_dir):
    # Create dataset directory
    annot_dir = os.path.join(dataset_dir, 'annotations')
    save_dir = os.path.join(annot_dir, 'coco_converted')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    default_json_file = os.path.join(annot_dir, 'instances_default.json')
    default_json_file = Path(default_json_file)
    result_json_file = os.path.join(annot_dir, 'instances_result.json')
    result_json_file = Path(result_json_file)

    lname = result_json_file.stem.replace("instances_", "")
    fn = save_dir / lname / "labels"  # folder name
    fn.mkdir(parents=True, exist_ok=True)
    with open(default_json_file) as f:
        coco_data = json.load(f)
    coco_data.pop('annotations')

    # Create image dict
    images = {f'{x["id"]:d}': x for x in coco_data["images"]}
    annotations = list()
    ann_id = 1

    # Read labels file
    for img_id, img in TQDM(images.items(), desc=f"Annotations {default_json_file}"):
        h, w = img["height"], img["width"]
        file_name:str = img["file_name"]
        npos = file_name.find('images/')
        if npos >= 0:
            file_name = file_name[npos+7:]

        # Read
        lb_file:Path = (fn / file_name).with_suffix(".txt")
        if lb_file.is_file():
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if len(lb) > 0:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                points = lb[:, 1:]
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"
            
            for label in lb:
                cat_id, point = int(label[0]), label[1:]
                tx, tw = point[[0,2]].astype(float) * w
                ty, th = point[[1,3]].astype(float) * h
                tx -= tw / 2
                ty -= th / 2
                bbox = [tx, ty, tw, th]
                ann = dict(id=ann_id, image_id=int(img_id), category_id=cat_id+1, bbox=bbox, area=tw*th,
                    iscrowd=0, segmentation=[])
                ann_id += 1
                annotations.append(ann)
    coco_data['annotations'] = annotations

    json_str = json.dumps(coco_data, ensure_ascii=False, indent=None)
    open(result_json_file, 'w').write(json_str)


def main(args):
    if args.work_mode == 'split':
        split_dataset(args.dataset_dir)
    elif args.work_mode == 'check':
        check_dataset(args.dataset_dir)
    elif args.work_mode == 'coco2yolo':
        convert_coco_to_yolo(args.dataset_dir)
    elif args.work_mode == 'yolo2coco':
        convert_yolo_to_coco(args.dataset_dir)
    return True

if __name__ == '__main__':
    args = parse_args()
    main(args)

