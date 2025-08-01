# File: scripts/step_05_evaluate.py

from ultralytics import YOLO
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

def evaluate_model():
    """在测试集上评估最终训练好的模型。"""
    print("\n--- 第5步: 在测试集上评估最终模型 ---")

    model_weights_path = os.path.join(settings.FINAL_MODEL_DIR, 'weights', 'best.pt')

    if not os.path.exists(model_weights_path):
        print(f"错误: 未找到模型权重文件 {model_weights_path}")
        return

    print(f"正在加载最终模型: {model_weights_path}")
    model = YOLO(model_weights_path)

    print("正在 'test' 数据集上进行评估...")
    metrics = model.val(
        data=os.path.join(settings.YOLO_DATASET_DIR, 'data.yaml'),
        split='test',
        device=settings.DEVICE
    )

    print("\n--- 最终评估结果 ---")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@75:    {metrics.box.map75:.4f}")
    print("✅ 评估完成！")

if __name__ == '__main__':
    evaluate_model()