# Run Test Integration Guide

这个目录包含了竞赛主办方提供的测试框架，用于验证你的模型提交是否符合要求。

## 目录结构

```
run_test/
 submit/              # 你的提交文件
    model/          # 训练好的模型文件
    src/            # 推理代码
       predictor.py
    requirements.txt
 src/                # 测试框架代码
 docker-compose.yml  # Docker环境配置
 Dockerfile
 run.py             # 测试执行脚本
 make_val.py        # 验证数据生成脚本
```

## 使用流程

### 1. 训练模型
在项目根目录进行模型训练：
```bash
python train.py
```

### 2. 准备提交文件
训练完成后，确保以下文件已正确设置：
- run_test/submit/model/best.pt - 训练好的YOLO模型
- run_test/submit/src/predictor.py - 推理代码（已自动更新）
- run_test/submit/requirements.txt - 依赖包列表（已自动更新）

### 3. 准备测试数据
确保数据集在正确位置：
```bash
# 数据集应该在项目根目录的dataset文件夹
dataset/
 annotations/
    train.json
 images/
     T1.jpg
     T2.jpg
     ...
```

### 4. 创建验证数据
```bash
cd run_test
python make_val.py --load-anno-path ../dataset/annotations/train.json --save-anno-path ../dataset/annotations/custom_val.json --val-ratio 0.25 --seed 42
```

### 5. 运行测试
```bash
cd run_test
python run.py --input-data-dir ../dataset --input-name annotations/custom_val.json
```

### 6. 使用Docker测试（推荐）
```bash
cd run_test
docker compose up -d
docker exec -it axell_2025 bash

# 在容器内
pip install -r requirements.txt
python run.py --input-data-dir ../dataset --input-name annotations/custom_val.json
```

## 注意事项

1. **模型文件**: 确保训练后的模型文件被正确复制到 run_test/submit/model/best.pt
2. **依赖管理**: submit目录下的requirements.txt用于最终提交，根目录的requirements.txt用于开发
3. **数据路径**: 所有路径都相对于run_test目录设置
4. **GPU支持**: Docker环境已配置GPU支持，确保你的系统有合适的NVIDIA驱动

## 提交前检查清单

- [ ] 模型文件存在且可加载
- [ ] predictor.py实现正确
- [ ] requirements.txt包含所有依赖
- [ ] 本地测试通过
- [ ] Docker环境测试通过
- [ ] 输出格式符合要求（最多5个检测框）
