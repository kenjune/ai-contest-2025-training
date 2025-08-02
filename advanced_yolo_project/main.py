# File: main.py

from scripts.step_01_deduplicate_images import find_and_remove_duplicates
from scripts.step_02_prepare_yolo_dataset import prepare_yolo_data
from scripts.step_03_tune_hyperparameters import tune_parameters
from scripts.step_04_train_final_model import train_final
from scripts.step_05_evaluate import evaluate_model

def run_pipeline():
    """按顺序执行从数据处理到模型评估的整个流程。"""
    # 第1步: 清洗数据集，移除相似图片，并清理内存
    #find_and_remove_duplicates()

    # 第2步: 分割并转换数据集为YOLO格式
    prepare_yolo_data()

    # 第3步: 使用W&B进行超参数调优
    tune_parameters()

    # 第4步: 使用找到的最佳参数训练最终模型
    train_final()

    # 第5步: 在测试集上评估最终模型的性能
    evaluate_model()

    print("\n🎉🎉🎉 整个流程已成功完成！🎉🎉🎉")

if __name__ == '__main__':
    run_pipeline()