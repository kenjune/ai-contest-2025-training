# File: scripts/step_02_prepare_yolo_dataset.py

import json
import os
import random
import shutil
import sys
from tqdm import tqdm

# 导入prepare_dataset脚本
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
    
    # 🔧 添加：返回 categories
    return coco_data['categories']


# --- 2. 调用外部脚本进行转换的函数 (优化了你的逻辑) ---

def _run_external_converter():
    """调用外部转换器的主要逻辑"""
    
    try:
        # 🔧 创建符合prepare_dataset期望的输入目录结构
        input_dir = settings.PREPARE_DATASET_INPUT_DIR
        
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        
        annotations_dir = os.path.join(input_dir, 'annotations')
        images_dir = os.path.join(input_dir, 'images')
        os.makedirs(annotations_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        print(f"\n--> 创建prepare_dataset输入目录: {input_dir}")
        
        # 🔧 复制分割后的标注文件到annotations/目录
        for split in ['train', 'val', 'test']:
            src_path = os.path.join(settings.SPLIT_ANNOTATIONS_DIR, f'{split}.json')
            dst_path = os.path.join(annotations_dir, f'{split}.json')
            shutil.copy(src_path, dst_path)
            print(f"    复制标注: {split}.json")
        
        # 🔧 收集所有需要的图片并复制到images/目录
        print("\n--> 收集需要的图片列表...")
        needed_images = set()
        
        for split in ['train', 'val', 'test']:
            json_path = os.path.join(annotations_dir, f'{split}.json')
            with open(json_path, 'r') as f:
                data = json.load(f)
            for img_info in data['images']:
                needed_images.add(img_info['file_name'])
        
        print(f"    需要复制 {len(needed_images)} 张图片")
        
        # 🔧 复制图片到images/目录（使用硬复制，确保Kaggle环境兼容）
        print("\n--> 复制图片文件...")
        copied_count = 0
        missing_count = 0
        missing_files = []
        
        for img_file in tqdm(needed_images, desc="复制图片"):
            src_path = os.path.join(settings.RAW_IMAGES_DIR, img_file)
            dst_path = os.path.join(images_dir, img_file)
            
            if os.path.exists(src_path):
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e:
                    print(f"    ❌ 复制失败 {img_file}: {e}")
                    missing_count += 1
            else:
                missing_files.append(img_file)
                missing_count += 1
        
        print(f"    ✅ 复制完成: {copied_count} 成功, {missing_count} 失败")
        
        if missing_count > 0:
            print(f"    ⚠️ 缺失文件前5个: {missing_files[:5]}")
        
        # 🔧 确保输出目录被清理
        if os.path.exists(settings.YOLO_DATASET_DIR):
            shutil.rmtree(settings.YOLO_DATASET_DIR)
            print(f"    清理现有输出目录: {settings.YOLO_DATASET_DIR}")
        
        # 🔧 调用prepare_dataset进行转换
        print(f"\n--> 调用prepare_dataset转换...")
        print(f"    输入: {input_dir}")
        print(f"    输出: {settings.YOLO_DATASET_DIR}")
        
        prepare_dataset_main(
            src_path=input_dir,              # prepare_dataset期望的输入结构
            dst_path=settings.YOLO_DATASET_DIR,  # YOLO格式输出目录
            force_remove=True                # 强制删除现有目录
        )
        
        print("    ✅ prepare_dataset转换完成")
        
        # 🔧 验证转换结果
        _verify_conversion_result()
        
    except Exception as e:
        print(f"❌ 转换过程出错: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # 🔧 清理临时输入目录
        if 'input_dir' in locals() and os.path.exists(input_dir):
            shutil.rmtree(input_dir)
            print(f"✅ 已清理临时目录: {input_dir}")

def _verify_conversion_result():
    """验证转换结果"""
    print("\n🔍 验证转换结果...")
    
    # 检查data.yaml
    data_yaml_path = os.path.join(settings.YOLO_DATASET_DIR, 'data.yaml')
    if os.path.exists(data_yaml_path):
        print("✅ data.yaml 已生成")
        with open(data_yaml_path, 'r') as f:
            content = f.read()
            print("📄 data.yaml内容:")
            print(content)
    else:
        print("❌ data.yaml 未找到")
    
    # 检查目录结构
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(settings.YOLO_DATASET_DIR, 'images', split)
        label_dir = os.path.join(settings.YOLO_DATASET_DIR, 'labels', split)
        
        if os.path.exists(img_dir):
            img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✅ images/{split}: {img_count} 张图片")
        else:
            print(f"❌ images/{split} 目录不存在")
            
        if os.path.exists(label_dir):
            label_count = len([f for f in os.listdir(label_dir) if f.endswith('.txt')])
            print(f"✅ labels/{split}: {label_count} 个标签")
        else:
            print(f"❌ labels/{split} 目录不存在")

def prepare_yolo_data():
    """主函数：执行完整的数据集准备流程。"""
    print("\n--- 第2步: 准备YOLO格式数据集 ---")
    
    try:
        # 🔧 修复：使用正确的函数名
        categories = _validate_and_split_annotations()  # 不是 _split_annotations()
        
        # 第二步：调用外部转换器
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