"""
数据集下载和设置脚本
运行此脚本来自动下载和设置项目数据集
"""

import os
import zipfile
import urllib.request
from pathlib import Path

def download_dataset():
    """下载数据集"""
    
    # 数据集配置
    DATASET_CONFIG = {
        'url': 'https://example.com/dataset.zip',  # 替换为实际的数据集下载链接
        'filename': 'dataset.zip',
        'extract_to': 'dataset/',
        'expected_size': 100 * 1024 * 1024  # 100MB，根据实际情况调整
    }
    
    print("🚀 开始下载数据集...")
    
    # 创建数据集目录
    dataset_dir = Path(DATASET_CONFIG['extract_to'])
    dataset_dir.mkdir(exist_ok=True)
    
    # 检查是否已经存在数据集
    if check_dataset_exists(dataset_dir):
        print("✅ 数据集已存在，跳过下载")
        return
    
    # 下载数据集
    try:
        print(f"📥 正在下载: {DATASET_CONFIG['url']}")
        urllib.request.urlretrieve(
            DATASET_CONFIG['url'], 
            DATASET_CONFIG['filename'],
            show_progress
        )
        
        # 解压数据集
        print("📂 正在解压数据集...")
        with zipfile.ZipFile(DATASET_CONFIG['filename'], 'r') as zip_ref:
            zip_ref.extractall(DATASET_CONFIG['extract_to'])
        
        # 删除压缩包
        os.remove(DATASET_CONFIG['filename'])
        
        print("✅ 数据集下载完成！")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("请手动下载数据集并放置在 dataset/ 目录下")

def show_progress(block_num, block_size, total_size):
    """显示下载进度"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\r进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')

def check_dataset_exists(dataset_dir):
    """检查数据集是否已存在"""
    required_files = [
        'annotations/train.json',
        'images/'
    ]
    
    for file_path in required_files:
        full_path = dataset_dir / file_path
        if not full_path.exists():
            return False
    return True

def setup_dataset_structure():
    """设置数据集目录结构"""
    
    print("📁 设置数据集目录结构...")
    
    # 创建必要的目录
    directories = [
        'dataset/annotations',
        'dataset/images',
        'dataset/train',
        'dataset/val',
        'models',
        'logs',
        'checkpoints'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

if __name__ == "__main__":
    print("🎯 AI Contest 2025 - 数据集设置")
    print("=" * 50)
    
    # 设置目录结构
    setup_dataset_structure()
    
    # 下载数据集
    download_dataset()
    
    print("\n" + "=" * 50)
    print("🎉 数据集设置完成！")
    print("\n📋 下一步:")
    print("1. 检查 dataset/ 目录下的文件")
    print("2. 运行 python prepare_dataset.py 进行数据预处理")
    print("3. 运行 python train.py 开始训练")
