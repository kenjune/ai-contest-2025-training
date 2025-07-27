# 个人开发工作流程指南

## 🚀 每天工作开始前 (重要!)

```bash
# 1. 切换到开发分支
git checkout develop

# 2. 拉取其他人的最新更改
git pull origin develop

# 3. 查看当前分支状态
git status
```

## 💡 开发新功能

### 创建功能分支
```bash
# 从 develop 创建新分支
git checkout -b feature/你的功能名

# 功能分支命名建议：
# feature/model-improvement    (模型改进)
# feature/data-processing      (数据处理)
# feature/evaluation-metrics   (评估指标)
# feature/config-optimization  (配置优化)
```

### 开发过程中的提交
```bash
# 添加文件
git add .

# 或者添加特定文件
git add config.py train.py

# 提交 (使用清晰的描述)
git commit -m "添加新的数据增强方法"

# 推送到远程
git push origin feature/你的功能名
```

## 📤 提交代码到团队

### 方式1: Pull Request (推荐)
1. 推送你的功能分支到 GitHub
2. 在 GitHub 网页上点击 "Compare & pull request"
3. **重要**: 确保目标分支是 `develop`，不是 `main`
4. 填写 PR 描述，说明你做了什么
5. 等待代码审查和合并

### 方式2: 直接合并 (小团队可用)
```bash
# 切换到 develop 分支
git checkout develop

# 拉取最新代码
git pull origin develop

# 合并你的功能分支
git merge feature/你的功能名

# 推送到远程
git push origin develop
```

## 🔄 功能完成后的清理

```bash
# 切换回 develop
git checkout develop

# 更新本地 develop 分支
git pull origin develop

# 删除本地功能分支
git branch -d feature/你的功能名

# 删除远程功能分支 (可选)
git push origin --delete feature/你的功能名
```

## ⚠️ 常见错误和解决方法

### 错误1: 直接在 develop 分支开发
```bash
# 错误做法
git checkout develop
# 直接修改代码并提交 ❌

# 正确做法
git checkout develop
git checkout -b feature/new-feature  # 创建新分支 ✅
# 在新分支开发
```

### 错误2: 向 main 分支提交
```bash
# 错误: Pull Request 目标选择了 main ❌
# 正确: Pull Request 目标应该是 develop ✅
```

### 错误3: 忘记拉取最新代码
```bash
# 每天开始前务必执行
git checkout develop
git pull origin develop  # 获取其他人的更新 ✅
```

## 🤝 协作最佳实践

### 提交信息规范
```bash
# 好的提交信息
git commit -m "添加YOLO模型的数据增强功能"
git commit -m "修复训练过程中的内存泄漏问题"
git commit -m "更新配置文件支持新的学习率调度器"

# 不好的提交信息
git commit -m "修改"        # 太简单 ❌
git commit -m "更新代码"     # 不具体 ❌
```

### 分支命名规范
```bash
# 功能开发
feature/model-optimization
feature/data-preprocessing
feature/ui-improvements

# 错误修复
bugfix/training-crash
bugfix/config-loading

# 紧急修复
hotfix/memory-leak
hotfix/security-patch
```

### 代码同步频率
- **每天开始工作前**: 必须拉取最新代码
- **功能开发中**: 建议每天推送一次
- **功能完成后**: 立即创建 Pull Request

## 📋 快速检查清单

开始工作前:
- [ ] `git checkout develop`
- [ ] `git pull origin develop`
- [ ] `git checkout -b feature/xxx`

提交代码时:
- [ ] `git add .`
- [ ] `git commit -m "清晰的描述"`
- [ ] `git push origin feature/xxx`
- [ ] 在 GitHub 创建 PR 到 develop

完成后:
- [ ] PR 被合并
- [ ] `git checkout develop`
- [ ] `git pull origin develop`
- [ ] `git branch -d feature/xxx`
