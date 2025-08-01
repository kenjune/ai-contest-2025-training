# File: scripts/step_04_train_final_model.py

from ultralytics import YOLO
import torch
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

def train_final():
    """使用调优找到的最佳超参数，训练最终的模型。"""
    print("\n--- 第4步: 训练最终模型 ---")

    if not os.path.exists(settings.BEST_PARAMS_FILE):
        print(f"错误: 未找到最佳参数文件 {settings.BEST_PARAMS_FILE}")
        return

    with open(settings.BEST_PARAMS_FILE, 'r') as f:
        best_hyperparameters = json.load(f)

    print("已加载最佳超参数用于最终训练:")
    print(json.dumps(best_hyperparameters, indent=2))
    best_hyperparameters['batch'] = 4 

    model = YOLO(settings.TUNING_MODEL)
    
    print("\n开始最终模型训练...")
    model.train(
        data=os.path.join(settings.YOLO_DATASET_DIR, 'data.yaml'),
        epochs=settings.FINAL_MODEL_EPOCHS,
        imgsz=settings.FINAL_MODEL_IMG_SIZE,
        device=settings.DEVICE,
        name='final_model_run',
        project=os.path.dirname(settings.FINAL_MODEL_DIR),
        exist_ok=True,
        **best_hyperparameters
    )
    print(f"\n✅ 最终模型训练完成！模型已保存在 '{settings.FINAL_MODEL_DIR}' 文件夹中。")

if __name__ == '__main__':
    train_final()