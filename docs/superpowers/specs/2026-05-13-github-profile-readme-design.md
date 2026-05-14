# GitHub Profile README 设计规格

## 概述

为 GitHub 用户 `3336-zheng` 创建一个创意向 + 技术栈展示的 Profile README。

- **风格**：极客动效流
- **主题色**：亮色系蓝色
- **外部服务依赖**：typing-svg、skillicons.dev、github-readme-stats、streak-stats、header.netlify.app

## 用户信息

| 字段 | 值 |
|------|-----|
| GitHub 用户名 | 3336-zheng |
| 展示名字 | 郑均皓（晚秋） |
| Slogan | 从机器学习到深度学习，从KWS到ASR，从工作流到Agent+RAG，探索AI的无限可能 |
| 邮箱 | 2868687158@qq.com |
| CSDN | https://blog.csdn.net/zhengjhzuishuai |

## 技术栈

Python、PyTorch、TensorFlow、Pandas、NumPy、Git、Docker、Linux、VSCode、MySQL、Redis

## 模块结构（自上而下）

### ① 打字机动画 Header

- 服务：`github-readme-typing-svg`
- 内容：两行循环 —— `晚秋_3336` 和 slogan
- 字体：`JetBrains Mono`
- 颜色：蓝色系（`#2196F3` → `#0D47A1`）
- 背景：透明

### ② 个人简介区

纯 Markdown，内容：

> Hi，我是 **郑均皓（晚秋）**，专注于端侧智能语音技术与 AI Agent 应用开发。
> 热爱探索 KWS、ASR、RAG、Agent 等前沿方向，正在用代码让机器更懂人。

### ③ 技术栈图标阵列

- 服务：`skillicons.dev`
- 图标：Python、PyTorch、TensorFlow、Pandas、NumPy、Git、Docker、Linux、VSCode、MySQL、Redis
- 点击可跳转对应官网

### ④ GitHub 统计卡片（两列并排）

- 服务：`github-readme-stats`
- 主题：`default`（亮色蓝色系）
- 左列：GitHub Stats（commits、PRs、stars、contributed to）
- 右列：Top Languages（条形图）
- 布局：HTML `<table>` 两列

### ⑤ 贡献 Streak 卡片

- 服务：`github-readme-streak-stats`
- 主题：`default`（亮色蓝色系）
- 显示：当前连续贡献天数、最长连续贡献天数、总贡献数

### ⑥ 精选项目

Markdown 表格，2 个项目：

| 项目 | 说明 | 技术栈 |
|------|------|--------|
| 智语 | 端侧智能语音笔记助手（自研 Agent + RAG） | Python, RAG, Agent, ASR |
| ArcFace-CNN-classifying_words-model_ONNX | 机器学习框架做分类识别词 | Python, ONNX, CNN |

### ⑦ 页脚

- 波浪动画：`header.netlify.app` 生成的 SVG 波浪
- 社交徽章：`shields.io` 风格
  - GitHub → https://github.com/3336-zheng
  - 邮箱 → mailto:2868687158@qq.com
  - CSDN → https://blog.csdn.net/zhengjhzuishuai
- 底部文字：`Thanks for visiting! 🚀`

## 外部服务 URL 模板

所有 URL 中的 `3336-zheng` 均替换为实际 GitHub 用户名。

```
# 打字机动画
https://readme-typing-svg.demolab.com/?font=JetBrains+Mono&weight=600&size=24&pause=1000&color=2196F3&center=true&vCenter=true&multiline=true&repeat=true&width=600&height=80&lines=%E6%99%9A%E7%A7%8B_3336;%E4%BB%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%88%B0%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%EF%BC%8C%E4%BB%8EKWS%E5%88%B0ASR%EF%BC%8C%E4%BB%8E%E5%B7%A5%E4%BD%9C%E6%B5%81%E5%88%B0Agent%2BRAG%EF%BC%8C%E6%8E%A2%E7%B4%A2AI%E7%9A%84%E6%97%A0%E9%99%90%E5%8F%AF%E8%83%BD

# 统计卡片
https://github-readme-stats.vercel.app/api?username=3336-zheng&show_icons=true&theme=default&locale=cn

# Top Languages
https://github-readme-stats.vercel.app/api/top-langs/?username=3336-zheng&layout=compact&theme=default&locale=cn

# Streak
https://github-readme-streak-stats.herokuapp.com/?user=3336-zheng&theme=default&locale=cn

# 技术栈图标
https://skillicons.dev/icons?i=python,pytorch,tensorflow,pandas,numpy,git,docker,linux,vscode,mysql,redis

# 波浪动画（蓝色系）
https://capsule-render.vercel.app/api?type=waving&color=2196F3&height=150&section=footer
```

## 输出文件

生成一个 `README.md` 文件，用户需将其推送到 `3336-zheng/3336-zheng` 仓库（GitHub Profile README 特殊仓库）。

## 依赖

无代码依赖。所有功能通过 Markdown + 外部服务 URL 实现。
