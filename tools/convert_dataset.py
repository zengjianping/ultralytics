import os, sys, math, copy
import shutil, time, json
import argparse, shutil
import cv2, glob
import numpy as np
from os import path as osp
from pathlib import Path
from xml.dom import minidom
from collections import defaultdict

work_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, work_dir)

from ultralytics.data.utils import check_det_dataset
from ultralytics.data.split import autosplit
from ultralytics.data.converter import convert_coco
from ultralytics.utils import TQDM


def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '/data/Datasets/ModelTrainData/airport/nanning'
    parser.add_argument('--work_mode', type=str, default='check')
    parser.add_argument('--dataset_dir', type=str, default=f'{data_dir}/d2_ir/job_20250522')
    parser.add_argument('--sub_datasets', nargs='*', help='One or more sub datasets')
    parser.add_argument('--class_names', type=str, help='class names')
    args = parser.parse_args()
    return args

def split_dataset(dataset_dir):
    image_dir = os.path.join(dataset_dir, 'images')
    autosplit(path=image_dir, weights=(0.8, 0.2, 0.0), annotated_only=False)

def check_dataset(dataset_dir):
    desc_file = os.path.join(dataset_dir, 'data.yaml')
    data = check_det_dataset(desc_file, autodownload=False)
    print(data)

def convert_voc_to_yolo(dataset_dir, class_names:str='person'):
    def convert_coordinates(size, box):
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        x = (box[0] + box[1]) / 2.0 * dw
        y = (box[2] + box[3]) / 2.0 * dh
        w = (box[1] - box[0]) * dw
        h = (box[3] - box[2]) * dh
        return (x,y,w,h)

    class_map = dict()
    for class_id, class_name in enumerate(class_names.split(',')):
        class_map[class_name] = class_id

    label_dir = os.path.join(dataset_dir, 'labels')
    xml_files = glob.glob(osp.join(label_dir, '*.xml'))

    for xml_file in TQDM(xml_files, desc=f"Annotations {label_dir}"):
        label_file = xml_file.replace('.xml', '.txt')
        wfile = open(label_file, "w")

        xmldoc = minidom.parse(xml_file)
        itemlist = xmldoc.getElementsByTagName('object')
        size = xmldoc.getElementsByTagName('size')[0]
        width = int((size.getElementsByTagName('width')[0]).firstChild.data)
        height = int((size.getElementsByTagName('height')[0]).firstChild.data)

        for item in itemlist:
            # get class label
            classid = (item.getElementsByTagName('name')[0]).firstChild.data
            if classid in class_map:
                label_str = str(class_map[classid])
            else:
                print("warning: label '%s' not in look-up table" % classid)
                continue

            # get bbox coordinates
            xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
            ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
            xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
            ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
            box = (float(xmin), float(xmax), float(ymin), float(ymax))
            box = convert_coordinates((width,height), box)
            wfile.write(label_str + " " + " ".join([("%.6f" % a) for a in box]) + '\n')

        wfile.close()


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
        with open(txt_file, "w") as file:
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

def repair_dataset(dataset_dir):
    for subset in ['train', 'val', 'test']:
        label_dir = os.path.join(dataset_dir, subset, 'labels')
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        for label_file in label_files:
            rfile = open(label_file, 'r', encoding='utf-8')
            labels = [x.split() for x in rfile.read().strip().splitlines() if len(x)]
            labels = np.array(labels, dtype=np.float32)
            rfile.close()
            if any(labels[:,0] == 1):
                print(subset, label_file)
                print(labels)
                wfile = open(label_file, "w", encoding='utf-8')
                for i in range(len(labels)):
                    box = [0] + labels[i,1:].tolist()
                    line = (*box,)  # cls, box or segments
                    wfile.write(("%g " * len(line)).rstrip() % line + "\n")

def format_dir2file(dataset_dir):
    dataset_path = Path(dataset_dir)
    for subset in ['train', 'val', 'test']:
        subset_dir = os.path.join(dataset_dir, subset)
        if os.path.exists(subset_dir):
            image_dir = os.path.join(subset_dir, 'images')
            image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
            image_files += glob.glob(os.path.join(image_dir, '*.png'))
            subset_file = os.path.join(dataset_dir, f'{subset}.txt')
            wfile = open(subset_file, "w", encoding='utf-8')
            for image_file in image_files:
                wfile.write(f"./{Path(image_file).relative_to(dataset_path).as_posix()}" + "\n")

def merge_datasets(dataset_dir, sub_datasets):
    if sub_datasets is None:
        return
    os.makedirs(dataset_dir, exist_ok=True)
    for subset in ['train', 'val', 'test']:
        subset_file = os.path.join(dataset_dir, f'{subset}.txt')
        yaml_file = os.path.join(dataset_dir, f'data.yaml')
        wfile = open(subset_file, "w", encoding='utf-8')
        for sub_dataset in sub_datasets:
            if not os.path.isfile(yaml_file):
                source_file = os.path.join(sub_dataset, f'data.yaml')
                shutil.copyfile(source_file, yaml_file)
            dataset_subset_file = os.path.join(sub_dataset, f'{subset}.txt')
            if os.path.isfile(dataset_subset_file):
                rfile = open(dataset_subset_file, 'r', encoding='utf-8')
                file_names = [x.strip() for x in rfile.read().strip().splitlines() if len(x)]
                file_names = [os.path.join(sub_dataset,x) for x in file_names if len(x)]
                file_names = [os.path.abspath(x) for x in file_names if len(x)]
                wfile.write('\n'.join(file_names) + '\n')
        wfile.close()

def main(args):
    if args.work_mode == 'split':
        split_dataset(args.dataset_dir)
    elif args.work_mode == 'check':
        check_dataset(args.dataset_dir)
    elif args.work_mode == 'voc2yolo':
        convert_voc_to_yolo(args.dataset_dir, args.class_names)
    elif args.work_mode == 'coco2yolo':
        convert_coco_to_yolo(args.dataset_dir)
    elif args.work_mode == 'yolo2coco':
        convert_yolo_to_coco(args.dataset_dir)
    elif args.work_mode == 'repair':
        repair_dataset(args.dataset_dir)
    elif args.work_mode == 'dir2file':
        format_dir2file(args.dataset_dir)
    elif args.work_mode == 'merge':
        merge_datasets(args.dataset_dir, args.sub_datasets)
    return True

if __name__ == '__main__':
    args = parse_args()
    main(args)

