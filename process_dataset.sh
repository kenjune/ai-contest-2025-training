#!/bin/bash
# 数据集处理示例脚本

echo "🎯 AI Contest 2025 - 数据集预处理"
echo "=================================="

# 检查是否存在源数据集
if [ ! -d "dataset_raw" ]; then
    echo "❌ 错误: 找不到 dataset_raw 目录"
    echo "请先下载并解压数据集到 dataset_raw/ 目录"
    exit 1
fi

# 检查源数据集结构
echo "📋 检查数据集结构..."
if [ ! -d "dataset_raw/annotations" ]; then
    echo "❌ 错误: 缺少 annotations 目录"
    exit 1
fi

if [ ! -d "dataset_raw/images" ]; then
    echo "❌ 错误: 缺少 images 目录"
    exit 1
fi

echo "✅ 数据集结构检查通过"

# 显示数据集信息
echo "📊 数据集信息:"
echo "  - 图片数量: $(find dataset_raw/images -name "*.jpg" -o -name "*.png" | wc -l)"
echo "  - 标注文件: $(find dataset_raw/annotations -name "*.json" | wc -l)"

# 执行数据预处理
echo "🔄 开始数据预处理..."
python prepare_dataset.py --src dataset_raw --dst dataset_processed --force

# 检查处理结果
if [ $? -eq 0 ]; then
    echo "✅ 数据预处理完成!"
    echo ""
    echo "📁 输出目录结构:"
    tree dataset_processed -L 2 2>/dev/null || find dataset_processed -type d | head -10
    echo ""
    echo "📄 生成的配置文件:"
    cat dataset_processed/data.yaml
    echo ""
    echo "🎉 现在可以开始训练了:"
    echo "    python train.py"
else
    echo "❌ 数据预处理失败，请检查错误信息"
    exit 1
fi
