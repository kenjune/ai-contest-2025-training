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

# --- 调试功能函数 ---

def _debug_paths():
    """调试所有关键路径"""
    print("\n🔍 === 详细路径调试 ===")
    
    # 1. 检查原始数据路径
    print(f"1️⃣ 原始数据路径:")
    print(f"   RAW_DATA_DIR: {settings.RAW_DATA_DIR}")
    print(f"   RAW_IMAGES_DIR: {settings.RAW_IMAGES_DIR}")
    print(f"   RAW_IMAGES_DIR exists: {'✅' if os.path.exists(settings.RAW_IMAGES_DIR) else '❌'}")
    
    if os.path.exists(settings.RAW_IMAGES_DIR):
        raw_files = [f for f in os.listdir(settings.RAW_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"   原始图片总数: {len(raw_files)}")
        print(f"   前5个文件: {raw_files[:5]}")
        
        # 检查T10.jpg是否存在
        if 'T10.jpg' in raw_files:
            print(f"   ✅ T10.jpg 在原始目录中存在")
            t10_path = os.path.join(settings.RAW_IMAGES_DIR, 'T10.jpg')
            t10_size = os.path.getsize(t10_path)
            print(f"   T10.jpg 大小: {t10_size} bytes")
        else:
            print(f"   ❌ T10.jpg 在原始目录中不存在")
            # 检查是否有类似名称
            similar_files = [f for f in raw_files if 'T10' in f or 't10' in f]
            print(f"   类似文件: {similar_files}")
    
    # 2. 检查处理后的路径
    print(f"\n2️⃣ 处理后的路径:")
    print(f"   PROCESSED_DATA_DIR: {settings.PROCESSED_DATA_DIR}")
    print(f"   SPLIT_ANNOTATIONS_DIR: {settings.SPLIT_ANNOTATIONS_DIR}")
    print(f"   YOLO_DATASET_DIR: {settings.YOLO_DATASET_DIR}")
    
    # 3. 检查分割后的JSON文件
    print(f"\n3️⃣ 分割后的JSON文件:")
    for split in ['train', 'val', 'test']:
        json_path = os.path.join(settings.SPLIT_ANNOTATIONS_DIR, f'{split}.json')
        print(f"   {split}.json exists: {'✅' if os.path.exists(json_path) else '❌'}")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            img_names = [img['file_name'] for img in data['images']]
            print(f"   {split} 包含图片: {len(img_names)}")
            
            # 检查T10.jpg在哪个分割中
            if 'T10.jpg' in img_names:
                print(f"   ✅ T10.jpg 在 {split} 分割中")

def _debug_image_copying(needed_images, src_dir, dst_dir):
    """调试图片复制过程"""
    print(f"\n🔍 === 图片复制调试 ===")
    print(f"需要复制的图片总数: {len(needed_images)}")
    print(f"源目录: {src_dir}")
    print(f"目标目录: {dst_dir}")
    
    # 检查T10.jpg的复制过程
    if 'T10.jpg' in needed_images:
        print(f"\n📋 T10.jpg 复制详情:")
        src_path = os.path.join(src_dir, 'T10.jpg')
        dst_path = os.path.join(dst_dir, 'T10.jpg')
        
        print(f"   源路径: {src_path}")
        print(f"   目标路径: {dst_path}")
        print(f"   源文件存在: {'✅' if os.path.exists(src_path) else '❌'}")
        
        if os.path.exists(src_path):
            src_size = os.path.getsize(src_path)
            print(f"   源文件大小: {src_size} bytes")
            
            # 尝试复制
            try:
                shutil.copy2(src_path, dst_path)
                print(f"   ✅ 复制成功")
                if os.path.exists(dst_path):
                    dst_size = os.path.getsize(dst_path)
                    print(f"   目标文件大小: {dst_size} bytes")
                    print(f"   大小匹配: {'✅' if src_size == dst_size else '❌'}")
                else:
                    print(f"   ❌ 复制后目标文件不存在")
            except Exception as e:
                print(f"   ❌ 复制失败: {e}")

def _debug_yolo_structure():
    """调试YOLO数据集结构"""
    print(f"\n🔍 === YOLO数据集结构调试 ===")
    
    yolo_dir = settings.YOLO_DATASET_DIR
    print(f"YOLO数据集目录: {yolo_dir}")
    print(f"目录存在: {'✅' if os.path.exists(yolo_dir) else '❌'}")
    
    if not os.path.exists(yolo_dir):
        return
    
    # 检查data.yaml
    data_yaml_path = os.path.join(yolo_dir, 'data.yaml')
    print(f"\ndata.yaml 存在: {'✅' if os.path.exists(data_yaml_path) else '❌'}")
    
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            content = f.read()
        print("data.yaml 内容:")
        print(content)
    
    # 检查每个分割的目录
    for split in ['train', 'val', 'test']:
        print(f"\n📁 {split} 分割:")
        
        img_dir = os.path.join(yolo_dir, 'images', split)
        label_dir = os.path.join(yolo_dir, 'labels', split)
        
        print(f"   images/{split} 存在: {'✅' if os.path.exists(img_dir) else '❌'}")
        print(f"   labels/{split} 存在: {'✅' if os.path.exists(label_dir) else '❌'}")
        
        if os.path.exists(img_dir):
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   图片数量: {len(img_files)}")
            
            # 检查T10.jpg
            if 'T10.jpg' in img_files:
                print(f"   ✅ T10.jpg 在 images/{split} 中")
                t10_path = os.path.join(img_dir, 'T10.jpg')
                t10_size = os.path.getsize(t10_path)
                print(f"   T10.jpg 大小: {t10_size} bytes")
                print(f"   T10.jpg 完整路径: {t10_path}")
            else:
                print(f"   ❌ T10.jpg 不在 images/{split} 中")
                # 显示前5个文件作为参考
                print(f"   前5个文件: {img_files[:5]}")

# --- 原有函数保持不变，但添加调试调用 ---

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
    """读取、验证、清洗并随机分割COCO标注数据。"""
    print("--> 1. 正在验证、清洗和分割标注文件...")
    
    # 🔧 修复：优先使用清洗后的文件，如果不存在则使用原始文件
    if os.path.exists(settings.CLEANED_ANNOTATION_FILE):
        annotation_file = settings.CLEANED_ANNOTATION_FILE
        print(f"🔍 使用清洗后的标注文件: {annotation_file}")
    else:
        # 使用原始标注文件
        original_annotation_file = os.path.join(settings.RAW_ANNOTATIONS_DIR, 'train.json')
        if os.path.exists(original_annotation_file):
            annotation_file = original_annotation_file
            print(f"🔍 使用原始标注文件: {annotation_file}")
            print("⚠️ 未找到清洗后的标注文件，将直接使用原始文件")
        else:
            print(f"❌ 既没有清洗后的标注文件，也没有原始标注文件")
            print(f"   清洗后文件: {settings.CLEANED_ANNOTATION_FILE}")
            print(f"   原始文件: {original_annotation_file}")
            raise FileNotFoundError("无法找到标注文件，请确保运行了 step_01 或原始标注文件存在")
    
    print(f"文件存在: {'✅' if os.path.exists(annotation_file) else '❌'}")
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    print(f"原始数据统计:")
    print(f"   图片数量: {len(coco_data['images'])}")
    print(f"   标注数量: {len(coco_data['annotations'])}")
    print(f"   类别数量: {len(coco_data['categories'])}")

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

    # 🔧 添加调试：检查T10.jpg
    t10_images = [img for img in valid_images if img['file_name'] == 'T10.jpg']
    if t10_images:
        print(f"🔍 T10.jpg 在有效图片中: ✅")
        print(f"   图片ID: {t10_images[0]['id']}")
        print(f"   尺寸: {t10_images[0]['width']}x{t10_images[0]['height']}")
    else:
        print(f"🔍 T10.jpg 在有效图片中: ❌")

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
    
    print(f"数据集分割结果:")
    for name, images in datasets.items():
        print(f"   {name}: {len(images)} 张图片")
        # 检查T10.jpg在哪个分割中
        t10_in_split = any(img['file_name'] == 'T10.jpg' for img in images)
        if t10_in_split:
            print(f"   🔍 T10.jpg 在 {name} 分割中")
    
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
    
    return coco_data['categories']

def _run_external_converter():
    """调用外部转换器的主要逻辑"""
    
    try:
        # 🔧 添加路径调试
        _debug_paths()
        
        # 创建符合prepare_dataset期望的输入目录结构
        input_dir = settings.PREPARE_DATASET_INPUT_DIR
        
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        
        annotations_dir = os.path.join(input_dir, 'annotations')
        images_dir = os.path.join(input_dir, 'images')
        os.makedirs(annotations_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        print(f"\n--> 创建prepare_dataset输入目录: {input_dir}")
        
        # 复制分割后的标注文件到annotations/目录
        for split in ['train', 'val', 'test']:
            src_path = os.path.join(settings.SPLIT_ANNOTATIONS_DIR, f'{split}.json')
            dst_path = os.path.join(annotations_dir, f'{split}.json')
            shutil.copy(src_path, dst_path)
            print(f"    复制标注: {split}.json")
        
        # 收集所有需要的图片并复制到images/目录
        print("\n--> 收集需要的图片列表...")
        needed_images = set()
        
        for split in ['train', 'val', 'test']:
            json_path = os.path.join(annotations_dir, f'{split}.json')
            with open(json_path, 'r') as f:
                data = json.load(f)
            for img_info in data['images']:
                needed_images.add(img_info['file_name'])
        
        print(f"    需要复制 {len(needed_images)} 张图片")
        
        # 🔧 添加图片复制调试
        _debug_image_copying(needed_images, settings.RAW_IMAGES_DIR, images_dir)
        
        # 复制图片到images/目录（使用硬复制，确保Kaggle环境兼容）
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
                    
                    # 🔧 验证复制结果
                    if not os.path.exists(dst_path):
                        print(f"    ⚠️ 复制后文件不存在: {img_file}")
                    
                except Exception as e:
                    print(f"    ❌ 复制失败 {img_file}: {e}")
                    missing_count += 1
            else:
                missing_files.append(img_file)
                missing_count += 1
                print(f"    ❌ 源文件不存在: {src_path}")
        
        print(f"    ✅ 复制完成: {copied_count} 成功, {missing_count} 失败")
        
        if missing_count > 0:
            print(f"    ⚠️ 缺失文件前10个: {missing_files[:10]}")
        
        # 确保输出目录被清理
        if os.path.exists(settings.YOLO_DATASET_DIR):
            shutil.rmtree(settings.YOLO_DATASET_DIR)
            print(f"    清理现有输出目录: {settings.YOLO_DATASET_DIR}")
        
        # 调用prepare_dataset进行转换
        print(f"\n--> 调用prepare_dataset转换...")
        print(f"    输入: {input_dir}")
        print(f"    输出: {settings.YOLO_DATASET_DIR}")
        
        prepare_dataset_main(
            src_path=input_dir,
            dst_path=settings.YOLO_DATASET_DIR,
            force_remove=True
        )
        
        print("    ✅ prepare_dataset转换完成")
        
        # 🔧 添加YOLO结构调试
        _debug_yolo_structure()
        
    except Exception as e:
        print(f"❌ 转换过程出错: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # 清理临时输入目录
        if 'input_dir' in locals() and os.path.exists(input_dir):
            shutil.rmtree(input_dir)
            print(f"✅ 已清理临时目录: {input_dir}")

def _verify_conversion_result():
    """验证转换结果"""
    print("\n🔍 验证转换结果...")
    _debug_yolo_structure()

def prepare_yolo_data():
    """主函数：执行完整的数据集准备流程。"""
    print("\n--- 第2步: 准备YOLO格式数据集 ---")
    
    try:
        # 第一步：分割标注文件
        categories = _validate_and_split_annotations()
        
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