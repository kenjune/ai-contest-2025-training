# prepare_dataset.py 使用指南

## 📖 脚本功能

`prepare_dataset.py` 是一个数据集预处理工具，主要功能：

1. **格式转换**: 将 COCO 格式转换为 Ultralytics YOLO 格式
2. **数据组织**: 重新整理文件结构，便于训练
3. **数据分割**: 自动创建训练/验证集分割
4. **配置生成**: 生成 YOLO 训练配置文件

## 🎯 使用场景

当你有 COCO 格式的数据集，但想使用 Ultralytics YOLO 进行训练时使用。

## 📁 输入数据格式要求

脚本期望的源数据集结构：
```
source_dataset/
├── annotations/
│   ├── train.json      # 训练集标注 (COCO格式)
│   ├── val.json        # 验证集标注 (可选)
│   └── test.json       # 测试集标注 (可选)
└── images/
    ├── T1.jpg
    ├── T2.jpg
    └── ...             # 所有图片文件
```

## 🚀 使用方法

### 基本使用
```bash
python prepare_dataset.py --src /path/to/coco_dataset --dst /path/to/yolo_dataset
```

### 强制覆盖已存在的输出目录
```bash
python prepare_dataset.py --src /path/to/coco_dataset --dst /path/to/yolo_dataset --force
```

### 本项目中的具体示例
```bash
# 假设你的 COCO 数据集在 dataset_raw/ 目录
python prepare_dataset.py --src dataset_raw --dst dataset_processed

# 如果需要重新处理
python prepare_dataset.py --src dataset_raw --dst dataset_processed --force
```

## 📤 输出结果

转换后的目录结构：
```
dataset_processed/
├── images/
│   ├── train/          # 训练图片
│   │   ├── T1.jpg
│   │   └── ...
│   ├── val/            # 验证图片
│   └── test/           # 测试图片 (如果有)
├── labels/
│   ├── train/          # 训练标注 (YOLO格式)
│   │   ├── T1.txt
│   │   └── ...
│   ├── val/            # 验证标注
│   └── test/           # 测试标注
└── data.yaml           # YOLO 配置文件
```

## ⚙️ 配置说明

在 `config.py` 中可以调整的参数：

### 数据分割比例
```python
TRAIN_SPLIT = 0.8  # 80% 用于训练，20% 用于验证
```

### 数据集标签
```python
LABELS = ("train", "val", "test")  # 处理的数据集类型
```

## 📝 生成的配置文件示例

`data.yaml` 文件内容示例：
```yaml
path: /absolute/path/to/dataset_processed  # 数据集根目录
train: images/train  # 训练图片路径
val: images/val      # 验证图片路径
test: images/test    # 测试图片路径

# 类别定义
names:
    0: person
    1: car
    2: bicycle
    # ... 其他类别
```

## 🔧 高级用法

### 自动验证集生成
如果源数据中没有 `val.json`，脚本会自动：
1. 从训练集中分割出验证集
2. 按照 `TRAIN_SPLIT` 比例分割 (默认 80:20)
3. 移动对应的图片和标注文件

### 符号链接 vs 文件复制
- **Linux/Mac**: 创建符号链接，节省磁盘空间
- **Windows**: 复制文件，确保兼容性

## ⚠️ 注意事项

### 路径要求
- 使用绝对路径，避免路径错误
- 确保有足够的磁盘空间

### 文件权限
- 确保源目录有读取权限
- 确保目标目录有写入权限

### 数据完整性
- 检查 COCO JSON 文件格式正确
- 确保图片文件存在且可读取

## 🐛 常见问题

### Q: 脚本报错 "Folder not found"
A: 检查源数据集路径是否正确，确保包含 `annotations/` 和 `images/` 目录

### Q: 转换后类别 ID 不对
A: COCO 格式类别 ID 从 1 开始，YOLO 从 0 开始，脚本会自动转换

### Q: 内存不足
A: 处理大数据集时可能内存不足，建议：
- 分批处理
- 增加虚拟内存
- 使用更强的机器

## 📊 性能优化

### 加速处理
- 使用 SSD 存储
- 确保充足内存
- 对于 Windows，考虑使用 WSL

### 磁盘空间优化
- Linux/Mac 系统会创建符号链接
- Windows 系统会复制文件，需要更多空间
