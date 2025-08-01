# File: config/settings.py

import os
import torch

# --- 项目结构路径 ---
# 获取项目的根目录绝对路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- 数据路径 (核心修改) ---
# !! 重要 !! 原始数据集的路径，这里我们使用你在Kaggle Notebook中的路径
# 如果你在本地运行，并且数据在别处，请修改这个路径
RAW_DATA_DIR = '/kaggle/input/ai-contest-2025-training/dataset'
RAW_IMAGES_DIR = os.path.join(RAW_DATA_DIR, 'images')
RAW_ANNOTATIONS_DIR = os.path.join(RAW_DATA_DIR, 'annotations')

# 存放所有处理后文件的本地路径
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

# --- 中间文件和最终产物的本地路径 ---
CLEANED_ANNOTATION_FILE = os.path.join(PROCESSED_DATA_DIR, 'annotations', 'train_clean.json')
SPLIT_ANNOTATIONS_DIR = os.path.join(PROCESSED_DATA_DIR, 'annotations', 'splits')
YOLO_DATASET_DIR = os.path.join(PROCESSED_DATA_DIR, 'yolo_dataset')
BEST_PARAMS_FILE = os.path.join(ROOT_DIR, 'config', 'best_params.json') # 存放最优参数
FINAL_MODEL_DIR = os.path.join(ROOT_DIR, 'models', 'final_model') # 存放最终训练好的模型


# --- 数据去重配置 ---
SIMILARITY_THRESHOLD = 0.95

# --- 数据集分割比例 ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# 测试集比例将自动计算

# --- W&B 超参数调优配置 ---
WANDB_PROJECT_NAME = "YOLOv11-Tuning-Local-Project" # 你在W&B上的项目名称
SWEEP_CONFIG = {
    'method': 'bayes',
    'metric': {'name': 'val_mAP50-95', 'goal': 'maximize'},
    'early_terminate': {'type': 'hyperband', 'min_iter': 5},
    'parameters': {
        'imgsz': {'value': 640},
        'lr0': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
        'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-3},
        'mixup': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3},
    }
}
TUNING_EPOCHS = 25      # 每次调优试验训练的轮数
TUNING_BATCH_SIZE = 8   # 调优时使用的批大小 (根据你的显存调整)
TUNING_MODEL = 'yolov8x.pt' # 用于调优的模型
SWEEP_RUN_COUNT = 15    # 总共进行多少次调优试验

# --- 最终模型训练配置 ---
FINAL_MODEL_EPOCHS = 100
FINAL_MODEL_IMG_SIZE = 640

# --- 通用配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"