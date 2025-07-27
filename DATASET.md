# 数据集说明

## 📊 数据集信息

### 数据集来源
- **比赛**: Axell AI Contest 2025
- **类型**: 物体检测数据集
- **格式**: COCO 格式

### 数据集规模
- **训练图片**: ~1000张
- **标注格式**: JSON (COCO格式)
- **类别数量**: [根据实际情况填写]
- **总大小**: ~XXX MB

## 📁 目录结构

下载和设置完成后，数据集目录结构应该如下：

```
dataset/
├── annotations/
│   ├── train.json          # 训练集标注
│   └── val.json           # 验证集标注 (如果有)
├── images/
│   ├── T1.jpg
│   ├── T2.jpg
│   └── ...                # 所有图片文件
├── train/                 # 处理后的训练数据 (可选)
└── val/                   # 处理后的验证数据 (可选)
```

## 🚀 快速开始

### 方式 1: 自动下载 (推荐)
```bash
python download_dataset.py
```

### 方式 2: 手动下载
1. 从比赛网站下载 `dataset.zip`
2. 解压到项目根目录，确保目录结构如上所示
3. 运行数据预处理：`python prepare_dataset.py`

## ⚙️ 数据集配置

在 `config.py` 中修改数据集路径：

```python
# 数据集配置
DATASET_PATH = "dataset/"
TRAIN_ANNOTATIONS = "dataset/annotations/train.json"
TRAIN_IMAGES = "dataset/images/"
```

## 🔧 数据预处理

运行预处理脚本：
```bash
python prepare_dataset.py
```

这将：
- 验证数据集完整性
- 转换为 YOLO 格式 (如果需要)
- 生成训练/验证分割
- 创建类别映射文件

## 📝 注意事项

### 文件权限
- 确保数据集文件有读取权限
- 图片文件格式支持：`.jpg`, `.jpeg`, `.png`

### 存储建议
- 数据集不会上传到 Git 仓库 (已在 .gitignore 中排除)
- 建议本地保留备份
- 大型数据集可考虑使用外部存储

### 团队协作
- 团队成员需要分别下载数据集
- 可以考虑使用云存储共享
- 或设置团队内部的下载链接

## 🔗 相关链接

- [比赛官网](https://signate.jp/competitions/xxx)
- [数据集说明](https://github.com/kenjune/ai-contest-2025-training/blob/main/TRAIN.md)
- [环境配置](https://github.com/kenjune/ai-contest-2025-training/blob/main/README.md)

## ❓ 常见问题

### Q: 数据集下载失败怎么办？
A: 1. 检查网络连接 2. 尝试手动下载 3. 联系项目维护者

### Q: 数据集格式不对怎么办？
A: 运行 `python prepare_dataset.py` 进行格式转换

### Q: 磁盘空间不够怎么办？
A: 1. 清理临时文件 2. 使用外部存储 3. 考虑数据集采样
