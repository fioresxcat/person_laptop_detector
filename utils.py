import pdb
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import shutil
import json
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon


def parse_txt(fp, img_size, idx2name):
    img_w, img_h = img_size
    with open(fp) as f:
        lines = f.readlines()
    boxes, names = [], []
    for line in lines:
        cl, x, y, w, h = [float(el) for el in line.strip().split()]   
        x1, x2 = x - w/2, x + w/2
        y1, y2 = y - h/2, y + h/2
        x1, x2 = int(x1 * img_w), int(x2 * img_w) 
        y1, y2 = int(y1 * img_h), int(y2 * img_h) 
        boxes.append((x1, y1, x2, y2)) 
        if idx2name is not None:
            names.append(idx2name[int(cl)])
        else:
            names.append('object')
    return boxes, names


def to_txt(txt_path, boxes, names, img_shape, class2idx={}):
    h, w = img_shape
    with open(txt_path, 'w') as f:
        for box, label in zip(boxes, names):
            class_id = class2idx[label]
            if class_id == -1:
                continue
            xmin, ymin, xmax, ymax = box
            x, y, box_w, box_h = (xmax + xmin)/2, (ymax + ymin)/2, xmax-xmin, ymax-ymin
            x, y, box_w, box_h = x/w, y/h, box_w/w, box_h/h
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            box_w = np.clip(box_w, 0, 1)
            box_h = np.clip(box_h, 0, 1)
            f.write(f'{class_id} {x} {y} {box_w} {box_h}\n')


def parse_txt_with_confs(fp, img_size, idx2name):
    img_w, img_h = img_size
    with open(fp) as f:
        lines = f.readlines()
    boxes, names, confs = [], [], []
    for line in lines:
        cl, conf, x, y, w, h = [float(el) for el in line.strip().split()]   
        x1, x2 = x - w/2, x + w/2
        y1, y2 = y - h/2, y + h/2
        x1, x2 = int(x1 * img_w), int(x2 * img_w) 
        y1, y2 = int(y1 * img_h), int(y2 * img_h) 
        boxes.append((x1, y1, x2, y2)) 
        if idx2name is not None:
            names.append(idx2name[int(cl)])
        else:
            names.append('object')
        confs.append(conf)
    return boxes, names, confs


def parse_xml(xml):
    root = ET.parse(xml).getroot()
    objs = root.findall('object')
    boxes, ymins, obj_names = [], [], []
    for obj in objs:
        obj_name = obj.find('name').text
        box = obj.find('bndbox')
        xmin = float(box.find('xmin').text)
        ymin = float(box.find('ymin').text)
        xmax = float(box.find('xmax').text)
        ymax = float(box.find('ymax').text)
        ymins.append(ymin)
        boxes.append([xmin, ymin, xmax, ymax])
        obj_names.append(obj_name)
    indices = np.argsort(ymins)
    boxes = [boxes[i] for i in indices]
    obj_names = [obj_names[i] for i in indices]
    return boxes, obj_names



def write_to_xml(boxes, labels, size, xml_path):
    w, h = size
    root = ET.Element('annotations')
    filename = ET.SubElement(root, 'filename')
    filename.text = Path(xml_path).stem + '.jpg'
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    for box, label in zip(boxes, labels):
        box = [int(el) for el in box]
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = label
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin, ymin = ET.SubElement(bndbox, 'xmin'), ET.SubElement(bndbox, 'ymin')
        xmax, ymax = ET.SubElement(bndbox, 'xmax'), ET.SubElement(bndbox, 'ymax')
        xmin.text, ymin.text, xmax.text, ymax.text = map(str, box)
    ET.ElementTree(root).write(xml_path)



def xml2txt(xp, out_txt_fp, img_shape, class2idx):
    boxes, labels = parse_xml(str(xp))
    to_txt(out_txt_fp, boxes, labels, img_shape, class2idx)
    print(f'Done converting {xp} to {out_txt_fp}')



def plot_yolo_label(ip, tp, save_path, ignore_classes):
    img = cv2.imread(str(ip))
    boxes, names = parse_txt(tp, img.shape[:2][::-1], idx2name={0: 'person', 1: 'laptop'})
    for bb, name in zip(boxes, names):
        if name in ignore_classes: continue
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
    cv2.imwrite(save_path, img)


def compute_boxes_iou(box1, box2):
    x1, y1, x2, y2 = box1
    poly1 = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    x1, y1, x2, y2 = box2
    poly2 = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    intersect = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersect / union
    max_overlap_ratio = intersect / min(poly1.area, poly2.area)
    return iou, max_overlap_ratio