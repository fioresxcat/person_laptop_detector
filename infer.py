from utils import *
from ultralytics import YOLO

class2idx = {'person': 0, 'laptop': 1}
idx2class = {v:k for k, v in class2idx.items()}

def infer():
    ckpt_path = 'runs/detect/laptop/exp0_only_laptop_dataset/weights/best.pt'
    model = YOLO(ckpt_path)

    src_dir = 'data/laptop/coco/'
    out_dir = 'temp/check_coco_anno'
    os.makedirs(out_dir, exist_ok=True)
    cnt = 0
    for ip in Path(src_dir).glob('*.jpg'):
        im = cv2.imread(str(ip))
        tp = ip.with_suffix('.txt')
        boxes, names = parse_txt(tp, im.shape[:2][::-1], idx2name=idx2class)
        if 'laptop' not in names:
            continue
        num_laptop = names.count('laptop')
        res = model.predict(source=str(ip), imgsz=640, conf=0.7, iou=0.3, save=False, save_txt=False, verbose=False)
        if res[0].boxes is None:
            continue
        pred_data = res[0].boxes.data.detach().cpu().numpy()
        if len(pred_data) != num_laptop:
            shutil.copy(ip, out_dir)
            shutil.copy(tp, out_dir)
            write_to_xml(boxes, names, im.shape[:2][::-1], os.path.join(out_dir, ip.stem+'.xml'))
            cnt += 1
            print(f'Count: {cnt}, {ip} has different num laptops: true: {num_laptop}, pred: {len(pred_data)}')        


def infer_short():
    model = YOLO('runs/detect/person_laptop/exp3_more_human_data/weights/best.pt')
    print('num params: ', sum(p.numel() for p in model.parameters()))
    src_dir = 'data/test_images/laptop1.png'
    model.predict(source=src_dir, conf=0.1, iou=0.7, save=True, save_txt=True, save_conf=True, verbose=True, classes=[1, 2])

if __name__ == '__main__':
    # infer()
    infer_short()