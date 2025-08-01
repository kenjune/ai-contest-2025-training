# File: scripts/step_03_tune_hyperparameters.py

import wandb
from ultralytics import YOLO
import torch
import os
import gc
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import settings

def tune_parameters():
    """使用W&B Sweeps寻找最优超参数，并自动保存结果。"""
    print("\n--- 第3步: 使用W&B进行超参数调优 ---")

    try:
        wandb.login()
    except Exception as e:
        print(f"无法登录W&B。请先在终端运行 'wandb login' 或设置 WANDB_API_KEY 环境变量。错误: {e}")
        return

    def train_trial():
        run = wandb.init()
        hyperparameters = dict(run.config)
        model = None
        try:
            model = YOLO(settings.TUNING_MODEL)
            results = model.train(
                data=os.path.join(settings.YOLO_DATASET_DIR, 'data.yaml'),
                device=settings.DEVICE, name=f'sweep_run_{run.name}',
                exist_ok=True, verbose=False, epochs=settings.TUNING_EPOCHS,
                batch=settings.TUNING_BATCH_SIZE, **hyperparameters
            )
        except Exception as e:
            print(f"一次调优训练失败: {e}")
        finally:
            if model is not None: del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            run.finish()

    print("初始化 W&B Sweep...")
    sweep_id = wandb.sweep(sweep=settings.SWEEP_CONFIG, project=settings.WANDB_PROJECT_NAME)
    print(f"Sweep已启动, ID为: {sweep_id}。将运行 {settings.SWEEP_RUN_COUNT} 次试验。")
    wandb.agent(sweep_id, function=train_trial, count=settings.SWEEP_RUN_COUNT)

    print("\nSweep结束。正在获取最佳超参数...")
    api = wandb.Api()
    sweep = api.sweep(f"{api.default_entity}/{settings.WANDB_PROJECT_NAME}/{sweep_id}")
    best_run = sweep.best_run()
    
    print(f"找到最佳运行: {best_run.name}，其 mAP50-95 为: {best_run.summary.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    best_params = {k: v for k, v in best_run.config.items() if k in settings.SWEEP_CONFIG['parameters']}
    
    print("最佳超参数组合:")
    print(json.dumps(best_params, indent=2))
        
    with open(settings.BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"最佳参数已保存至: {settings.BEST_PARAMS_FILE}")

if __name__ == '__main__':
    tune_parameters()