# AI Contest 2025 - 训练项目

基于 [Ultralytics](https://github.com/ultralytics/ultralytics) 的 YOLO11 物体检测模型训练项目。

## 项目结构

```
├── config.py              # 配置文件
├── prepare_dataset.py      # 数据集预处理工具
├── requirements.txt        # 依赖包列表
├── train.py               # Python脚本训练
├── train.ipynb            # Jupyter Notebook训练
├── train_colab.ipynb      # Google Colab训练
├── TRAIN.md               # 原始训练文档
└── submit/                # 提交文件夹
    ├── requirements.txt
    ├── model/
    └── src/
        └── predictor.py
```

## 环境要求

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- 其他依赖见 `requirements.txt`

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

```bash
python prepare_dataset.py
```

### 3. 开始训练

#### 方式1: Python脚本
```bash
python train.py
```

#### 方式2: Jupyter Notebook
```bash
jupyter notebook train.ipynb
```

#### 方式3: Google Colab
上传 `train_colab.ipynb` 到 Google Colab 运行

## 分支管理

本项目采用以下分支结构进行多人协作：

- `main`: 主分支，包含稳定版本
- `develop`: 开发分支，用于功能集成
- `feature/*`: 功能分支，用于开发新功能
- `hotfix/*`: 热修复分支，用于紧急修复

## 贡献指南

1. Fork 本仓库
2. 创建功能分支: `git checkout -b feature/your-feature`
3. 提交变更: `git commit -am 'Add some feature'`
4. 推送分支: `git push origin feature/your-feature`
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证。
