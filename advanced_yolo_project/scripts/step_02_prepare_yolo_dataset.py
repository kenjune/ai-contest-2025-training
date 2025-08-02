# File: scripts/step_02_prepare_yolo_dataset.py (使用本地prepare_dataset脚本)

import json
import os
import random
import shutil
import sys
from tqdm import tqdm

# 导入本地的prepare_dataset脚本
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prepare_dataset import main as prepare_dataset_main
from config import settings

def _is_valid_bbox(bbox):
    """检查bbox是否有效"""
    if not bbox or len(bbox) != 4:
        return False
    
    # 检查所有值都不是None，并且都是数字
    for val in bbox:
        if val is None or not isinstance(val, (int, float)):
            return False
        if val < 0:  # bbox坐标不应该是负数
            return False
    
    # 检查宽度和高度大于0
    width, height = bbox[2], bbox[3]
    if width <= 0 or height <= 0:
        return False
    
    return True

def _split_annotations():
    """内部函数：将清洗后的COCO标注文件分割为train, val, test三个JSON文件。"""
    print("--> 1. 正在分割标注文件...")
    with open(settings.CLEANED_ANNOTATION_FILE, 'r') as f:
        coco_data = json.load(f)

    # 创建图片ID到图片名的映射
    image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

    # --- 标注过滤 ---
    original_annotations_count = len(coco_data['annotations'])
    
    valid_annotations = []
    invalid_count = 0
    
    for ann in coco_data['annotations']:
        # 检查必需字段和bbox有效性
        if ('image_id' not in ann or 
            'category_id' not in ann or 
            'bbox' not in ann or 
            not ann['bbox'] or 
            not _is_valid_bbox(ann['bbox'])):
            invalid_count += 1
            continue
        
        valid_annotations.append(ann)
    
    if invalid_count > 0:
        print(f"    移除了 {invalid_count} 条无效标注")
    
    coco_data['annotations'] = valid_annotations
    print(f"    保留了 {len(valid_annotations)} 条有效标注进行后续处理。")
    
    # 检查图片是否还有对应的标注
    image_ids_with_annotations = {ann['image_id'] for ann in valid_annotations}
    original_images = coco_data['images'][:]
    valid_images = [img for img in original_images if img['id'] in image_ids_with_annotations]
    
    if len(valid_images) != len(original_images):
        removed_count = len(original_images) - len(valid_images)
        print(f"    移除了 {removed_count} 张没有有效标注的图片。")
    
    coco_data['images'] = valid_images

    # 数据集分割
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
        print(f"    已生成 {name}.json ({len(subset_anns)} 条标注)")

    create_subset_json('train', train_images)
    create_subset_json('val', val_images)
    create_subset_json('test', test_images)
    
    return categories

def _prepare_dataset_input_structure():
    """准备prepare_dataset脚本需要的输入结构"""
    print("--> 2. 正在准备prepare_dataset输入结构...")
    
    # prepare_dataset期望的输入结构:
    # input_dir/
    #   ├── annotations/
    #   │   ├── train.json
    #   │   ├── val.json
    #   │   └── test.json
    #   └── images/
    #       └── (所有图片文件)
    
    input_dir = os.path.join(settings.PROCESSED_DATA_DIR, 'prepare_dataset_input')
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    
    # 创建目录结构
    annotations_dir = os.path.join(input_dir, 'annotations')
    images_dir = os.path.join(input_dir, 'images')
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # 复制标注文件
    for split in ['train', 'val', 'test']:
        src_file = os.path.join(settings.SPLIT_ANNOTATIONS_DIR, f'{split}.json')
        dst_file = os.path.join(annotations_dir, f'{split}.json')
        shutil.copy(src_file, dst_file)
        print(f"    已复制 {split}.json 到输入目录")
    
    # 创建图片的符号链接或复制图片
    print("    正在准备图片文件...")
    for img_file in tqdm(os.listdir(settings.RAW_IMAGES_DIR)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            src_path = os.path.join(settings.RAW_IMAGES_DIR, img_file)
            dst_path = os.path.join(images_dir, img_file)
            
            if sys.platform == "win32":
                # Windows下复制文件
                shutil.copy(src_path, dst_path)
            else:
                # 其他系统创建符号链接
                os.symlink(src_path, dst_path)
    
    print(f"    输入结构已准备完成: {input_dir}")
    return input_dir

def _call_prepare_dataset(input_dir):
    """调用本地的prepare_dataset脚本"""
    print("--> 3. 正在调用prepare_dataset脚本...")
    
    # 清理输出目录
    if os.path.exists(settings.YOLO_DATASET_DIR):
        shutil.rmtree(settings.YOLO_DATASET_DIR)
    
    try:
        # 直接调用prepare_dataset的main函数
        # main(src_path: str, dst_path: str, force_remove: bool = False)
        prepare_dataset_main(
            src_path=input_dir,
            dst_path=settings.YOLO_DATASET_DIR,
            force_remove=True
        )
        print("    ✅ prepare_dataset脚本执行成功！")
        return True
        
    except Exception as e:
        print(f"    ❌ prepare_dataset脚本执行失败: {e}")
        print("    详细错误信息:")
        import traceback
        traceback.print_exc()
        return False

def _cleanup_temp_files(input_dir):
    """清理临时文件"""
    print("--> 4. 正在清理临时文件...")
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
        print("    临时文件已清理")

def _verify_yolo_output():
    """验证YOLO输出"""
    print("--> 5. 正在验证YOLO格式输出...")
    
    required_dirs = [
        os.path.join(settings.YOLO_DATASET_DIR, 'images', 'train'),
        os.path.join(settings.YOLO_DATASET_DIR, 'images', 'val'),
        os.path.join(settings.YOLO_DATASET_DIR, 'images', 'test'),
        os.path.join(settings.YOLO_DATASET_DIR, 'labels', 'train'),
        os.path.join(settings.YOLO_DATASET_DIR, 'labels', 'val'),
        os.path.join(settings.YOLO_DATASET_DIR, 'labels', 'test'),
    ]
    
    data_yaml_path = os.path.join(settings.YOLO_DATASET_DIR, 'data.yaml')
    
    # 检查目录结构
    all_dirs_exist = True
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"    ❌ 缺少目录: {dir_path}")
            all_dirs_exist = False
        else:
            file_count = len(os.listdir(dir_path))
            split_name = os.path.basename(os.path.dirname(dir_path)) + '/' + os.path.basename(dir_path)
            print(f"    ✅ {split_name}: {file_count} 个文件")
    
    # 检查data.yaml
    if os.path.exists(data_yaml_path):
        print(f"    ✅ data.yaml 已创建: {data_yaml_path}")
    else:
        print(f"    ❌ data.yaml 缺失: {data_yaml_path}")
        all_dirs_exist = False
    
    return all_dirs_exist

def prepare_yolo_data():
    """主函数：执行数据集分割和YOLO格式转换。"""
    print("\n--- 第2步: 准备YOLO格式数据集 ---")
    
    try:
        # 第一步：分割标注文件
        categories = _split_annotations()
        
        # 第二步：准备prepare_dataset所需的输入结构
        input_dir = _prepare_dataset_input_structure()
        
        # 第三步：调用prepare_dataset脚本
        success = _call_prepare_dataset(input_dir)
        
        # 第四步：清理临时文件
        _cleanup_temp_files(input_dir)
        
        # 第五步：验证输出
        if success and _verify_yolo_output():
            print(f"\n✅ YOLO数据集已准备就绪，位于: {settings.YOLO_DATASET_DIR}")
            print("🎉 使用本地prepare_dataset脚本完成转换，完全避免了None值问题")
        else:
            print("\n❌ YOLO数据集准备过程中出现问题")
            return False
            
    except Exception as e:
        print(f"\n❌ 数据集准备失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    prepare_yolo_data()