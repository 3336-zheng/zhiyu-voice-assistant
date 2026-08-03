# 构建 React 前端
FROM node:22-alpine AS frontend-builder

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/vite.config.js ./
COPY frontend/app/ app/
RUN npm run build

# 智语端侧智能语音笔记助手运行镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（ffmpeg 用于音频处理，libmagic 用于文件类型检测）
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libmagic1 && \
    rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存（依赖没变就不重新安装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY main.py .
COPY backend/ backend/
COPY frontend/index.html frontend/summary.html frontend/docs.html frontend/style.css frontend/
COPY --from=frontend-builder /frontend/dist frontend/dist

# 创建数据目录（运行时通过 volume 挂载）
RUN mkdir -p data/database data/uploads data/wiki/pages data/wiki/attachments data/wiki/exports data/logs

# 暴露端口
EXPOSE 8337

# 启动命令
CMD ["python", "main.py"]
