# File: scripts/step_02_prepare_yolo_dataset.py (遵从你的要求，调用外部脚本的版本)

import json
import os
import random
import shutil
import sys
from tqdm import tqdm

# 导入你仓库根目录下的 prepare_dataset.py 脚本中的 main 函数
# 请确保你的 prepare_dataset.py 和 config.py 在 advanced_yolo_project 根目录下
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from prepare_dataset import main as prepare_dataset_main
except ImportError:
    print("错误：无法从上一级目录找到 'prepare_dataset.py'。请确保该文件存在于项目根目录中。")
    sys.exit(1)

from config import settings

# --- 1. 数据验证与分割函数 (保留了你优秀的校验逻辑) ---

def _is_valid_bbox(bbox):
    """检查bbox是否有效（坐标非负，宽高大于0）。"""
    if not bbox or len(bbox) != 4:
        return False
    if any(val is None or not isinstance(val, (int, float)) for val in bbox):
        return False
    x, y, w, h = bbox
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return False
    return True

def _validate_and_split_annotations():
    """读取、验证、清洗并随机分割清理后的COCO标注数据。"""
    print("--> 1. 正在验证、清洗和分割标注文件...")
    with open(settings.CLEANED_ANNOTATION_FILE, 'r') as f:
        coco_data = json.load(f)

    # 标注过滤
    original_annotations_count = len(coco_data['annotations'])
    valid_annotations = [ann for ann in coco_data['annotations'] if _is_valid_bbox(ann.get('bbox'))]
    
    removed_ann_count = original_annotations_count - len(valid_annotations)
    if removed_ann_count > 0:
        print(f"    - 已移除 {removed_ann_count} 条无效标注。")
    
    # 图片过滤 (只保留还有有效标注的图片)
    image_ids_with_annotations = {ann['image_id'] for ann in valid_annotations}
    original_images = coco_data['images']
    valid_images = [img for img in original_images if img['id'] in image_ids_with_annotations]

    removed_img_count = len(original_images) - len(valid_images)
    if removed_img_count > 0:
        print(f"    - 已移除 {removed_img_count} 张没有有效标注的图片。")
    
    print(f"    - 清洗后保留 {len(valid_images)} 张图片和 {len(valid_annotations)} 条标注。")

    # 数据集分割
    random.shuffle(valid_images)
    total = len(valid_images)
    train_count = int(total * settings.TRAIN_RATIO)
    val_count = int(total * settings.VAL_RATIO)

    datasets = {
        'train': valid_images[:train_count],
        'val': valid_images[train_count:train_count + val_count],
        'test': valid_images[train_count + val_count:]
    }
    
    # 创建并保存分割后的JSON文件
    os.makedirs(settings.SPLIT_ANNOTATIONS_DIR, exist_ok=True)
    annotations_by_image_id = {}
    for ann in valid_annotations:
        img_id = ann['image_id']
        annotations_by_image_id.setdefault(img_id, []).append(ann)

    for name, image_subset in datasets.items():
        image_ids = {img['id'] for img in image_subset}
        subset_anns = [ann for img_id in image_ids if img_id in annotations_by_image_id for ann in annotations_by_image_id[img_id]]
        subset_data = {'images': image_subset, 'annotations': subset_anns, 'categories': coco_data['categories']}
        path = os.path.join(settings.SPLIT_ANNOTATIONS_DIR, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(subset_data, f)
        print(f"    - 已生成 {name}.json")


# --- 2. 调用外部脚本进行转换的函数 (优化了你的逻辑) ---

def _run_external_converter():
    """
    创建临时目录结构，调用外部的prepare_dataset.py脚本，然后清理。
    """
    print("\n--> 2. 正在准备并调用外部转换脚本...")
    
    # 定义临时输入目录，作为prepare_dataset.py的--src参数
    temp_input_dir = os.path.join(settings.PROCESSED_DATA_DIR, 'temp_conversion_input')
    
    try:
        # --- a. 准备临时目录结构 ---
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
        
        temp_annotations_dir = os.path.join(temp_input_dir, 'annotations')
        temp_images_dir = os.path.join(temp_input_dir, 'images')
        os.makedirs(temp_annotations_dir)
        os.makedirs(temp_images_dir)
        
        # 将分割好的 aplit_annotations/*.json 复制到临时目录
        shutil.copytree(settings.SPLIT_ANNOTATIONS_DIR, temp_annotations_dir, dirs_exist_ok=True)
        
        # 将原始图片链接到临时目录 (使用symlink以节省空间和时间)
        print("    - 正在为图片创建临时链接...")
        all_raw_images = [f for f in os.listdir(settings.RAW_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for filename in tqdm(all_raw_images, desc="      - 链接图片"):
            os.symlink(
                os.path.join(settings.RAW_IMAGES_DIR, filename),
                os.path.join(temp_images_dir, filename)
            )
        print("    - 临时输入目录准备完毕。")

        # --- b. 调用外部脚本 ---
        print("\n--> 3. 正在执行 prepare_dataset.py...")
        prepare_dataset_main(
            src_path=temp_input_dir,
            dst_path=settings.YOLO_DATASET_DIR,
            force_remove=True
        )
        print("    - ✅ prepare_dataset.py 执行成功！")

    finally:
        # --- c. 清理工作 ---
        print("\n--> 4. 正在清理临时文件...")
        # 清理临时的split_annotations文件夹
        if os.path.exists(settings.SPLIT_ANNOTATIONS_DIR):
            shutil.rmtree(settings.SPLIT_ANNOTATIONS_DIR)
        # 清理为转换脚本创建的临时输入文件夹
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
        print("    - 清理完成。")

# --- 主执行函数 ---

def prepare_yolo_data():
    """主函数：执行完整的数据集准备流程。"""
    print("\n--- 第2步: 准备YOLO格式数据集 ---")
    
    try:
        # 验证、清洗并分割标注，生成临时的 train.json, val.json, test.json
        _validate_and_split_annotations()
        
        # 创建临时目录并调用外部脚本来完成转换
        _run_external_converter()
        
        print(f"\n✅ YOLO数据集已准备就绪，位于: {settings.YOLO_DATASET_DIR}")
        return True
        
    except Exception as e:
        print(f"\n❌ 数据集准备过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    prepare_yolo_data()