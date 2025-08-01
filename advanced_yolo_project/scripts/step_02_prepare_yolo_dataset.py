# File: scripts/step_02_prepare_yolo_dataset.py (增强版本 - 记录无效数据)

import json
import os
import random
import shutil
import sys
from tqdm import tqdm
from ultralytics.data.converter import convert_coco

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

    # --- 增强的标注过滤 - 记录无效数据 ---
    original_annotations_count = len(coco_data['annotations'])
    
    # 记录无效数据的详细信息
    invalid_data_info = {
        'no_bbox': {'count': 0, 'images': set()},
        'empty_bbox': {'count': 0, 'images': set()},
        'invalid_bbox': {'count': 0, 'images': set()},
        'missing_image_id': {'count': 0, 'images': set()},
        'missing_category_id': {'count': 0, 'images': set()}
    }
    
    valid_annotations = []
    
    for ann in coco_data['annotations']:
        image_name = "unknown_image"
        if 'image_id' in ann and ann['image_id'] in image_id_to_name:
            image_name = image_id_to_name[ann['image_id']]
        
        # 检查必需字段
        if 'image_id' not in ann:
            invalid_data_info['missing_image_id']['count'] += 1
            invalid_data_info['missing_image_id']['images'].add(image_name)
            continue
        
        if 'category_id' not in ann:
            invalid_data_info['missing_category_id']['count'] += 1
            invalid_data_info['missing_category_id']['images'].add(image_name)
            continue
            
        # 检查bbox
        if 'bbox' not in ann:
            invalid_data_info['no_bbox']['count'] += 1
            invalid_data_info['no_bbox']['images'].add(image_name)
            continue
            
        if not ann['bbox']:
            invalid_data_info['empty_bbox']['count'] += 1
            invalid_data_info['empty_bbox']['images'].add(image_name)
            continue
            
        if not _is_valid_bbox(ann['bbox']):
            invalid_data_info['invalid_bbox']['count'] += 1
            invalid_data_info['invalid_bbox']['images'].add(image_name)
            continue
        
        valid_annotations.append(ann)
    
    # 详细报告过滤结果
    total_invalid = original_annotations_count - len(valid_annotations)
    if total_invalid > 0:
        print(f"    检测到并移除了 {total_invalid} 条无效标注：")
        all_invalid_images = set()
        
        for reason, info in invalid_data_info.items():
            if info['count'] > 0:
                print(f"      - {reason}: {info['count']} 条标注")
                all_invalid_images.update(info['images'])
        
        print(f"    涉及的图片总数: {len(all_invalid_images)} 张")
    
    coco_data['annotations'] = valid_annotations
    print(f"    保留了 {len(valid_annotations)} 条有效标注进行后续处理。")
    
    # 检查图片是否还有对应的标注
    image_ids_with_annotations = {ann['image_id'] for ann in valid_annotations}
    original_images = coco_data['images'][:]
    valid_images = [img for img in original_images if img['id'] in image_ids_with_annotations]
    
    removed_images = []
    if len(valid_images) != len(original_images):
        removed_images = [img['file_name'] for img in original_images if img['id'] not in image_ids_with_annotations]
        print(f"    移除了 {len(removed_images)} 张没有有效标注的图片。")
    
    coco_data['images'] = valid_images
    
    # 在函数结束前返回无效数据信息
    return coco_data, invalid_data_info, removed_images

def _prepare_dataset_splits(coco_data):
    """准备数据集分割"""
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
        return categories

    create_subset_json('train', train_images)
    create_subset_json('val', val_images)
    return create_subset_json('test', test_images)

def _convert_to_yolo(categories):
    """内部函数：将分割后的COCO JSON文件转换为YOLOv8格式。"""
    print("\n--> 2. 正在转换为YOLO格式...")

    if os.path.exists(settings.YOLO_DATASET_DIR):
        shutil.rmtree(settings.YOLO_DATASET_DIR)

    # 在转换前再次验证数据
    print("    验证分割后的标注文件...")
    conversion_issues = []
    
    for split in ["train", "val", "test"]:
        json_path = os.path.join(settings.SPLIT_ANNOTATIONS_DIR, f"{split}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 验证每个标注
        invalid_annotations = []
        for ann in data['annotations']:
            if not _is_valid_bbox(ann.get('bbox')):
                # 找到对应的图片名
                image_name = "unknown"
                for img in data['images']:
                    if img['id'] == ann.get('image_id'):
                        image_name = img['file_name']
                        break
                invalid_annotations.append({
                    'image': image_name,
                    'annotation_id': ann.get('id', 'unknown'),
                    'bbox': ann.get('bbox')
                })
        
        if invalid_annotations:
            print(f"    警告: {split}.json 中仍有 {len(invalid_annotations)} 条无效标注！")
            conversion_issues.extend(invalid_annotations)
        else:
            print(f"    {split}.json 验证通过 ({len(data['annotations'])} 条标注)")

    # 转换标注文件
    print("    正在转换 train/val/test 的标注...")
    try:
        convert_coco(labels_dir=settings.SPLIT_ANNOTATIONS_DIR, use_segments=False, save_dir=settings.YOLO_DATASET_DIR)
        print("    COCO到YOLO转换成功！")
    except Exception as e:
        print(f"    转换过程中出现错误: {e}")
        if conversion_issues:
            print("    发现的问题标注:")
            for issue in conversion_issues[:10]:  # 只显示前10个
                print(f"      图片: {issue['image']}, bbox: {issue['bbox']}")
        raise
    
    # 复制图片
    print("    正在复制图片到YOLO目录结构...")
    copy_issues = []
    
    for split in ["train", "val", "test"]:
        json_path = os.path.join(settings.SPLIT_ANNOTATIONS_DIR, f"{split}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        image_dir = os.path.join(settings.YOLO_DATASET_DIR, 'images', split)
        os.makedirs(image_dir, exist_ok=True)
        
        copied_count = 0
        for img_info in tqdm(data['images'], desc=f"    复制 {split} 图片"):
            src_path = os.path.join(settings.RAW_IMAGES_DIR, img_info['file_name'])
            dst_path = os.path.join(image_dir, img_info['file_name'])
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                copied_count += 1
            else:
                copy_issues.append(img_info['file_name'])
        
        print(f"    {split} 集: 成功复制 {copied_count}/{len(data['images'])} 张图片")
    
    # 报告复制问题
    if copy_issues:
        print(f"    警告: {len(copy_issues)} 张图片未找到源文件")

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
            f.write(f"  {cat_id-1}: {dict_categories[cat_id]}\n")
    
    print(f"    data.yaml 已创建: {yaml_file_path}")
    return copy_issues

def prepare_yolo_data():
    """主函数：执行数据集分割和YOLO格式转换。"""
    print("\n--- 第2步: 准备YOLO格式数据集 ---")
    
    # 第一步：分割标注并记录无效数据
    coco_data, invalid_data_info, removed_images = _split_annotations()
    
    # 第二步：准备数据集分割
    categories = _prepare_dataset_splits(coco_data)
    
    # 第三步：转换为YOLO格式并记录问题
    copy_issues = _convert_to_yolo(categories)
    
    print(f"\nYOLO数据集已准备就绪，位于: {settings.YOLO_DATASET_DIR}")
    
    # ===========================================
    # 最终报告：汇总所有无效数据和问题
    # ===========================================
    print("\n" + "="*60)
    print("📊 数据质量报告")
    print("="*60)
    
    # 1. 无效标注统计
    total_invalid_annotations = sum(info['count'] for info in invalid_data_info.values())
    if total_invalid_annotations > 0:
        print(f"\n🔍 发现无效标注: {total_invalid_annotations} 条")
        for reason, info in invalid_data_info.items():
            if info['count'] > 0:
                print(f"  • {reason}: {info['count']} 条")
                print(f"    涉及图片 ({len(info['images'])} 张): {sorted(list(info['images']))[:5]}{'...' if len(info['images']) > 5 else ''}")
    else:
        print("\n✅ 所有标注数据有效")
    
    # 2. 移除的图片
    if removed_images:
        print(f"\n🗑️  移除图片: {len(removed_images)} 张")
        print(f"   图片列表: {removed_images[:10]}{'...' if len(removed_images) > 10 else ''}")
    else:
        print("\n✅ 所有图片都有有效标注")
    
    # 3. 复制问题
    if copy_issues:
        print(f"\n⚠️  图片复制问题: {len(copy_issues)} 张")
        print(f"   缺失文件: {copy_issues[:10]}{'...' if len(copy_issues) > 10 else ''}")
    else:
        print("\n✅ 所有图片复制成功")
    
    # 4. 汇总所有问题图片
    all_problem_images = set()
    for info in invalid_data_info.values():
        all_problem_images.update(info['images'])
    all_problem_images.update(removed_images)
    all_problem_images.update(copy_issues)
    
    if all_problem_images:
        print(f"\n📋 问题图片汇总: {len(all_problem_images)} 张")
        problem_list = sorted(list(all_problem_images))
        print(f"   完整列表: {problem_list[:15]}{'...' if len(problem_list) > 15 else ''}")
        
        # 保存问题图片列表到文件
        problem_file = os.path.join(settings.PROCESSED_DATA_DIR, 'problem_images.txt')
        os.makedirs(os.path.dirname(problem_file), exist_ok=True)
        with open(problem_file, 'w', encoding='utf-8') as f:
            f.write("问题图片列表\n")
            f.write("="*50 + "\n")
            for img in problem_list:
                f.write(f"{img}\n")
        print(f"   详细列表已保存到: {problem_file}")
    else:
        print("\n🎉 所有图片和标注都正常！")
    
    print("="*60)

if __name__ == '__main__':
    prepare_yolo_data()