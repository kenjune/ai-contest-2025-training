@echo off
REM Windows 版本的数据集处理脚本

echo 🎯 AI Contest 2025 - 数据集预处理
echo ==================================

REM 检查是否存在源数据集
if not exist "dataset_raw" (
    echo ❌ 错误: 找不到 dataset_raw 目录
    echo 请先下载并解压数据集到 dataset_raw\ 目录
    pause
    exit /b 1
)

REM 检查源数据集结构
echo 📋 检查数据集结构...
if not exist "dataset_raw\annotations" (
    echo ❌ 错误: 缺少 annotations 目录
    pause
    exit /b 1
)

if not exist "dataset_raw\images" (
    echo ❌ 错误: 缺少 images 目录
    pause
    exit /b 1
)

echo ✅ 数据集结构检查通过

REM 显示数据集信息
echo 📊 数据集信息:
for /f %%i in ('dir dataset_raw\images\*.jpg /b /s ^| find /c /v ""') do echo   - JPG图片数量: %%i
for /f %%i in ('dir dataset_raw\images\*.png /b /s ^| find /c /v ""') do echo   - PNG图片数量: %%i
for /f %%i in ('dir dataset_raw\annotations\*.json /b /s ^| find /c /v ""') do echo   - 标注文件: %%i

REM 执行数据预处理
echo 🔄 开始数据预处理...
python prepare_dataset.py --src dataset_raw --dst dataset_processed --force

REM 检查处理结果
if %errorlevel% equ 0 (
    echo ✅ 数据预处理完成!
    echo.
    echo 📁 输出目录结构:
    dir dataset_processed /s /b | findstr /r "dataset_processed\\[^\\]*$"
    echo.
    echo 📄 生成的配置文件:
    type dataset_processed\data.yaml
    echo.
    echo 🎉 现在可以开始训练了:
    echo     python train.py
) else (
    echo ❌ 数据预处理失败，请检查错误信息
    pause
    exit /b 1
)

pause
