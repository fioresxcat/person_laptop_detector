from utils import *

with open('coco_classes.txt') as f:
    coco_classes = [line.strip() for line in f.readlines()]
idx2name = {i:cl for i, cl in enumerate(coco_classes)}

def get_laptop_data_from_coco():
    img_dir = '/nfs-data2/tungtx2/class_surveillance/data/COCO/images/train2017'
    out_dir = 'data/laptop/coco'
    cnt = 0
    for ip in Path(img_dir).glob('*.jpg'):
        im = cv2.imread(str(ip))
        h, w, c = im.shape
        tp = str(ip).replace('/images/', '/labels/')
        tp = Path(tp).with_suffix('.txt')
        if not tp.exists():
            continue
        boxes, names = parse_txt(tp, (w, h), idx2name)
        if 'laptop' in names:
            new_boxes, new_names = [], []
            for box, name in zip(boxes, names):
                if name in ['laptop', 'person']:
                    new_boxes.append(box)
                    new_names.append(name)
            boxes, names = new_boxes, new_names
            
            save_dir = os.path.join(out_dir, 'images')
            os.makedirs(save_dir, exist_ok=True)
            shutil.copy(ip, save_dir)
            
            save_dir = os.path.join(out_dir, 'labels')
            os.makedirs(save_dir, exist_ok=True)
            to_txt(os.path.join(save_dir, tp.name), boxes, names, (h, w), class2idx={'person': 0, 'laptop': 1})
            cnt += 1
            print(f'done {cnt} files: {ip}')
            

def get_laptop_data_from_laptop_dataset():
    img_dir = '/nfs-data2/tungtx2/class_surveillance/data/laptop_dataset/train'
    out_dir = 'data/laptop/laptop_dataset'
    os.makedirs(out_dir, exist_ok=True)
    for ip in Path(img_dir).glob('*.jpg'):
        im = cv2.imread(str(ip))
        xp = ip.with_suffix('.xml')
        boxes, names = parse_xml(xp)
        names = ['laptop'] * len(names)
        save_dir = os.path.join(out_dir, 'images')
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy(ip, save_dir)
        save_dir = os.path.join(out_dir, 'labels')
        os.makedirs(save_dir, exist_ok=True)
        to_txt(os.path.join(save_dir, xp.stem + '.txt'), boxes, names, im.shape[:2], class2idx={'person': 0, 'laptop': 1})
        print(f'done {ip}')


def select_image_from_crowdhuman():
    img_dir = '/nfs-data2/tungtx2/class_surveillance/data/CrowdHuman/crowdhuman0102val-vbox-xml'
    pred_dir = '/nfs-data2/tungtx2/class_surveillance/train_detector/runs/detect/predict8/labels'
    out_dir = 'data/crowdhuman-vbox-no_laptop-10k'
    os.makedirs(out_dir, exist_ok=True)
    cnt = 0
    fpaths = list(Path(img_dir).glob('*.jpg'))
    for _ in range(10): np.random.shuffle(fpaths)
    for ip in fpaths:
        pred_tp = os.path.join(pred_dir, ip.stem+'.txt')
        if not os.path.exists(pred_tp):
            shutil.copy(ip, out_dir)
            shutil.copy(ip.with_suffix('.xml'), out_dir)
            cnt += 1
            print(f'done {cnt}: {ip} to {out_dir}')
            if cnt >= 10000:
                break
    
    print(f'total: {cnt}')

def xml2txt_dir():
    dir = 'data/crowdhuman-vbox-no_laptop-10k'
    for ip in Path(dir).rglob('*.jpg'):
        im = cv2.imread(str(ip))
        h, w = im.shape[:2]
        xp = ip.with_suffix('.xml')
        xml2txt(xp, xp.with_suffix('.txt'), (h, w), class2idx={'person': 0, 'head': 1, 'laptop': 2})



def combine_person_and_laptop_for_laptop_dataset():
    src_dir = '/nfs-data2/tungtx2/class_surveillance/train_detector/data/laptop/laptop_dataset'
    pred_dir = '/nfs-data2/tungtx2/class_surveillance/yolov5/runs/detect/exp7/labels'
    for ip in Path(src_dir).glob('*.jpg'):
        im = cv2.imread(str(ip))
        h, w, _ = im.shape
        xp = ip.with_suffix('.xml')
        xboxes, xnames = parse_xml(xp)
        pred_tp = os.path.join(pred_dir, ip.stem+'.txt')
        if os.path.exists(pred_tp):
            pboxes, pnames = parse_txt(pred_tp, (w, h), idx2name={0:'person'})
            all_boxes, all_names = xboxes + pboxes, xnames + pnames
            write_to_xml(all_boxes, all_names, (w, h), xp)
            print(f'DONE {ip}')


def txt2xml_dir():
    dir = 'data/crowdhuman-640x640-vbox'
    for ip in Path(dir).glob('*.jpg'):
        im = cv2.imread(str(ip))
        h, w = im.shape[:2]
        tp = ip.with_suffix('.txt')
        boxes, names = parse_txt(tp, (w, h), idx2name={0: 'head', 1: 'person'})
        out_xp = ip.with_suffix('.xml')
        write_to_xml(boxes, names, (w, h), out_xp)
        print(f'done {ip}')


def get_non_person_image_from_laptop_dataset():
    dir = 'data/laptop/laptop_dataset'
    pred_dir = '/nfs-data2/tungtx2/class_surveillance/yolov5/runs/detect/exp8/labels'
    out_dir = 'temp/laptop_ds_without_person'
    os.makedirs(out_dir, exist_ok=True)
    for ip in Path(dir).glob('*.jpg'):
        im = cv2.imread(str(ip))
        h, w, _ = im.shape
        xp = ip.with_suffix('.xml')
        pred_tp = os.path.join(pred_dir, ip.stem+'.txt')
        if not os.path.exists(pred_tp):
            shutil.copy(ip, out_dir)
            shutil.copy(xp, out_dir)
            print(f'done {ip}')


def split_train_val_random(dir, out_dir, val_ratio=0.15, seed=None):
    fpaths = list(Path(dir).glob('*.jpg'))
    np.random.seed(seed)
    for _ in range(10):
        np.random.shuffle(fpaths)
    num_train = int(len(fpaths) * (1-val_ratio))
    for index, fp in enumerate(fpaths):
        if index < num_train:
            split = 'train'
        else:
            split = 'val'
        save_dir = os.path.join(out_dir, split)
        os.makedirs(save_dir, exist_ok=True)
        shutil.move(str(fp), save_dir)

        xp = fp.with_suffix('.xml')
        if xp.exists(): shutil.move(str(xp), save_dir)

        tp = fp.with_suffix('.txt')
        if tp.exists(): shutil.move(str(tp), save_dir)

        jp = fp.with_suffix('.json')
        if jp.exists(): shutil.move(str(jp), save_dir)

        print(f'done {fp} to {save_dir}')


def merge_head_detection_into_xml():
    dir = '/nfs-data2/tungtx2/class_surveillance/train_detector/temp/laptop_ds_with_person_head'
    for ip in Path(dir).rglob('*.jpg'):
        im = cv2.imread(str(ip))
        h, w, _ = im.shape
        xp = ip.with_suffix('.xml')
        if not xp.exists(): continue
        boxes, names = parse_xml(xp)
        tp = ip.with_suffix('.txt')
        if not tp.exists(): continue
        head_boxes, head_names = parse_txt(tp, img_size=(w, h), idx2name={1: 'head'})
        remove_indexes = []
        for i, head_bb in enumerate(head_boxes):
            is_valid = False
            for bb, name in zip(boxes, names):
                if name != 'person': continue
                iou, max_overlap_ratio = compute_boxes_iou(head_bb, bb)
                if max_overlap_ratio > 0.8:
                    is_valid = True
                    break
            if not is_valid:
                remove_indexes.append(i)
        head_boxes = [bb for i, bb in enumerate(head_boxes) if i not in remove_indexes]
        head_names = ['head' for bb in head_boxes]
        boxes.extend(head_boxes)
        names.extend(head_names)
        write_to_xml(boxes, names, (w, h), xp)
        print(f'done write to {xp} with {len(head_boxes)} heads')



def nothing():
    dir = 'data/laptop/laptop_dataset_processed'
    bb_cnt, image_cnt = 0, 0
    for ip in Path(dir).rglob('*.jpg'):
        image_cnt += 1
        xp = ip.with_suffix('.xml')
        boxes, names = parse_xml(xp)
        if len(names) > 0:
            bb_cnt += names.count('laptop')
    print('image count: ', image_cnt)
    print('bb count: ', bb_cnt)
    print('mean bb per image: ', bb_cnt / image_cnt)


if __name__ == '__main__':
    # get_laptop_data_from_coco()
    # plot_yolo_label(
    #     '/nfs-data2/tungtx2/class_surveillance/train_detector/data/crowdhuman-640x640-vbox-no_laptop/train/273271,1a02900084ed5ae8.jpg',
    #     '/nfs-data2/tungtx2/class_surveillance/train_detector/data/crowdhuman-640x640-vbox-no_laptop/train/273271,1a02900084ed5ae8.txt',
    #     'a.jpg',
    #     ignore_classes=[]
    # )
    # get_laptop_data_from_laptop_dataset()
    nothing()
    # xml2txt_dir()
    # combine_person_and_laptop_for_laptop_dataset()
    # get_non_person_image_from_laptop_dataset()
    # split_train_val_random(
    #     dir='/nfs-data2/tungtx2/class_surveillance/train_detector/data/crowdhuman-vbox-no_laptop-10k',
    #     out_dir='/nfs-data2/tungtx2/class_surveillance/train_detector/data/crowdhuman-vbox-no_laptop-10k',
    # )
    # txt2xml_dir()
    # select_image_from_crowdhuman()
    # merge_head_detection_into_xml()