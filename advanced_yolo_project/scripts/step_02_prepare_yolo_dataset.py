# File: scripts/step_02_prepare_yolo_dataset.py
# File: scripts/step_02_prepare_yolo_dataset.py

import json
import os
import random
import shutil
import sys
from tqdm import tqdm
from ultralytics.data.converter import convert_coco

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

def _split_annotations():
    """内部函数：将清洗后的COCO标注文件分割为train, val, test三个JSON文件。"""
    print("--> 1. 正在分割标注文件...")
    with open(settings.CLEANED_ANNOTATION_FILE, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    annotations_by_image_id = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(ann)
        
    random.shuffle(images)

    total = len(images)
    train_count = int(total * settings.TRAIN_RATIO)
    val_count = int(total * settings.VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    print(f"    数据集分割完成：")
    print(f"    - 训练集: {len(train_images)} 张图片")
    print(f"    - 验证集: {len(val_images)} 张图片")
    print(f"    - 测试集: {len(test_images)} 张图片")
    
    os.makedirs(settings.SPLIT_ANNOTATIONS_DIR, exist_ok=True)

    def create_subset_json(name, image_subset):
        image_ids = {img['id'] for img in image_subset}
        subset_anns = []
        for img_id in image_ids:
            if img_id in annotations_by_image_id:
                subset_anns.extend(annotations_by_image_id[img_id])
                
        subset_data = {'images': image_subset, 'annotations': subset_anns, 'categories': categories}
        path = os.path.join(settings.SPLIT_ANNOTATIONS_DIR, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(subset_data, f)
        print(f"    已生成 {name}.json")
        return categories

    create_subset_json('train', train_images)
    create_subset_json('val', val_images)
    return create_subset_json('test', test_images)

def _convert_to_yolo(categories):
    """内部函数：将分割后的COCO JSON文件转换为YOLOv8格式。"""
    print("\n--> 2. 正在转换为YOLO格式...")

    if os.path.exists(settings.YOLO_DATASET_DIR):
        shutil.rmtree(settings.YOLO_DATASET_DIR)

    # 转换标注文件
    print("    正在转换 train/val/test 的标注...")
    convert_coco(labels_dir=settings.SPLIT_ANNOTATIONS_DIR, use_segments=False, save_dir=settings.YOLO_DATASET_DIR)
    
    # 复制/链接图片
    print("    正在复制图片到YOLO目录结构...")
    for split in ["train", "val", "test"]:
        json_path = os.path.join(settings.SPLIT_ANNOTATIONS_DIR, f"{split}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        image_dir = os.path.join(settings.YOLO_DATASET_DIR, 'images', split)
        os.makedirs(image_dir, exist_ok=True)
        
        for img_info in tqdm(data['images'], desc=f"    复制 {split} 图片"):
            src_path = os.path.join(settings.RAW_IMAGES_DIR, img_info['file_name'])
            dst_path = os.path.join(image_dir, img_info['file_name'])
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

    # 创建data.yaml配置文件
    print("    正在创建 data.yaml 配置文件...")
    dict_categories = {cat["id"]: cat["name"] for cat in categories}
    
    yaml_file_path = os.path.join(settings.YOLO_DATASET_DIR, "data.yaml")
    with open(yaml_file_path, mode="w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(settings.YOLO_DATASET_DIR)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("# Classes\n")
        f.write("names:\n")
        for cat_id in sorted(dict_categories.keys()):
            f.write(f"  {cat_id-1}: {dict_categories[cat_id]}\n") # YOLO从0开始

def prepare_yolo_data():
    """主函数：执行数据集分割和YOLO格式转换。"""
    print("\n--- 第2步: 准备YOLO格式数据集 ---")
    categories = _split_annotations()
    _convert_to_yolo(categories)
    print(f"YOLO数据集已准备就绪，位于: {settings.YOLO_DATASET_DIR}")

if __name__ == '__main__':
    prepare_yolo_data()