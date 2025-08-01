# File: scripts/step_01_deduplicate_images.py
# File: scripts/step_01_deduplicate_images.py

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cosine
import json
import sys
import gc

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

def find_and_remove_duplicates():
    """
    查找语义相似的图片，生成干净的标注文件，并清理内存。
    """
    print("--- 第1步: 图像去重与内存清理 ---")

    # 定义所有可能在此函数中创建的变量
    model, embeddings_dict, embeddings, files_to_remove = None, None, None, None
    
    try:
        # --- 1. 配置和准备 ---
        device = torch.device(settings.DEVICE)
        print(f"使用设备: {device}")
        os.makedirs(os.path.dirname(settings.CLEANED_ANNOTATION_FILE), exist_ok=True)

        # 加载预训练的ResNet50模型
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval().to(device)

        preprocess = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- 2. 提取图像特征向量 ---
        embeddings_dict = {}
        print("\n正在生成图像特征向量...")
        image_files = [f for f in os.listdir(settings.RAW_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
        with torch.no_grad():
            for filename in tqdm(image_files, desc="提取特征"):
                filepath = os.path.join(settings.RAW_IMAGES_DIR, filename)
                try:
                    img = Image.open(filepath).convert("RGB")
                    img_t = preprocess(img)
                    batch_t = torch.unsqueeze(img_t, 0).to(device)
                    embedding = model(batch_t)
                    embeddings_dict[filepath] = embedding.squeeze().cpu().numpy()
                except Exception as e:
                    print(f"\n无法处理文件 {filepath}: {e}")
        print(f"\n已为 {len(embeddings_dict)} 张图片生成特征向量。")

        # --- 3. 查找重复项 ---
        filepaths = list(embeddings_dict.keys())
        embeddings = np.array(list(embeddings_dict.values()))
        files_to_remove = set()
        visited = set()
        print("\n正在对比特征向量以查找重复图片...")
        for i in tqdm(range(len(filepaths)), desc="查找重复项"):
            if filepaths[i] in visited: continue
            for j in range(i + 1, len(filepaths)):
                if filepaths[j] in visited: continue
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                if similarity > settings.SIMILARITY_THRESHOLD:
                    files_to_remove.add(os.path.basename(filepaths[j]))
                    visited.add(filepaths[j])
        print(f"\n找到 {len(files_to_remove)} 张需要移除的重复图片。")

        # --- 4. 创建清洗后的标注文件 ---
        original_annotations_path = os.path.join(settings.RAW_ANNOTATIONS_DIR, 'train.json')
        print(f"正在创建清洗后的标注文件: {settings.CLEANED_ANNOTATION_FILE}")
        with open(original_annotations_path, 'r') as f:
            coco_data = json.load(f)

        cleaned_images = [img for img in coco_data['images'] if img['file_name'] not in files_to_remove]
        kept_image_ids = {img['id'] for img in cleaned_images}
        cleaned_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in kept_image_ids]
        cleaned_coco_data = {'images': cleaned_images, 'annotations': cleaned_annotations, 'categories': coco_data['categories']}
        
        with open(settings.CLEANED_ANNOTATION_FILE, 'w') as f:
            json.dump(cleaned_coco_data, f, indent=4)
            
        print(f"原始图片数量: {len(coco_data['images'])}")
        print(f"清洗后图片数量: {len(cleaned_images)}")
        print("图像去重完成。")

    finally:
        # --- 5. 清理内存和显存 ---
        print("\n开始清理ResNet相关对象和GPU缓存...")
        del model
        del embeddings_dict
        del embeddings
        del files_to_remove
        gc.collect()
        print("- 已执行垃圾回收。")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("- 已清空CUDA缓存。")
        print("清理完成！")


if __name__ == '__main__':
    find_and_remove_duplicates()