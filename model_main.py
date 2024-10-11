import os

import pdb
from ultralytics import YOLO

project = 'runs'
name = 'detect/person_laptop/exp3_more_human_data'
config_path = 'configs/person_laptop.yaml'
best_weights = os.path.join(os.getcwd(), project, name, 'weights/best.pt')

def train():
    model = YOLO('yolov9s.pt')
    model.train(data=config_path, epochs=100, batch=128, imgsz=640, cache='ram', device=[2,3,4,5], workers=64, exist_ok=True,
                optimizer='AdamW', seed=42, lr0=0.001, val=True, project=project, name=name,
                degrees=4, translate=0.1, scale=0.3, shear=0.1, fliplr=0.3, hsv_s=0.4, close_mosaic=10, mosaic=0.5,)


def validate():
    model = YOLO(best_weights)
    model.val(data=config_path, batch=32, split='test', imgsz=640)


def export():
    model = YOLO(best_weights)
    model.export(format='onnx', imgsz=640, dynamic=True, opset=14, simplify=True, half=False)


if __name__ == '__main__':
    train()
