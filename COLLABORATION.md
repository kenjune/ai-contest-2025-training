# 协作开发指南

## 仓库信息
- **主仓库**: https://github.com/kenjune/ai-contest-2025-training
- **开发分支**: https://github.com/kenjune/ai-contest-2025-training/tree/develop

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/kenjune/ai-contest-2025-training.git
cd ai-contest-2025-training
```

### 2. 切换到开发分支
```bash
git checkout develop
```

### 3. 查看分支状态
```bash
git branch -a
```

## 开发工作流

### 创建新功能分支
```bash
# 确保在最新的 develop 分支
git checkout develop
git pull origin develop

# 创建新功能分支
git checkout -b feature/你的功能名称
```

### 常用功能分支命名：
- `feature/model-optimization` - 模型优化
- `feature/data-preprocessing` - 数据预处理
- `feature/evaluation-metrics` - 评估指标
- `feature/config-improvements` - 配置改进
- `feature/documentation` - 文档更新

### 提交和推送代码
```bash
# 添加变更
git add .

# 提交变更
git commit -m "详细描述你的更改"

# 推送到远程
git push origin feature/你的功能名称
```

### 创建 Pull Request
1. 在 GitHub 上访问仓库
2. 点击 "Compare & pull request" 按钮
3. 确保目标分支是 `develop`
4. 填写 PR 描述
5. 点击 "Create pull request"

## 分支管理规则

### 分支说明
- **main**: 主分支，稳定版本，仅通过 PR 合并
- **develop**: 开发分支，功能集成，日常开发基础
- **feature/***: 功能分支，每个新功能一个分支
- **hotfix/***: 紧急修复分支，用于生产环境问题

### 合并流程
1. 功能开发 → `feature/xxx` 分支
2. 功能完成 → PR 到 `develop` 分支
3. 代码审查 → 合并到 `develop` 分支
4. 版本发布 → PR 从 `develop` 到 `main`

## 协作注意事项

### 代码同步
```bash
# 每天开始工作前，同步最新代码
git checkout develop
git pull origin develop

# 如果在功能分支，同步到功能分支
git checkout feature/你的功能名称
git merge develop
```

### 冲突解决
```bash
# 如果有冲突，手动解决后
git add 冲突文件
git commit -m "Resolve conflicts"
git push origin feature/你的功能名称
```

### 代码规范
- 提交信息使用中文，描述清晰
- 每个 commit 只做一件事
- 大功能拆分为多个小 commit
- PR 前确保代码能正常运行

## 联系方式
如有问题，请通过以下方式联系：
- GitHub Issues: https://github.com/kenjune/ai-contest-2025-training/issues
- 项目负责人: @kenjune
