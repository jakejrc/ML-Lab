<<<<<<< Updated upstream
FROM python:3.11-slim

LABEL maintainer="jake_jrc@qq.com"
LABEL org.opencontainers.image.title="ML-Lab"
LABEL org.opencontainers.image.description="Interactive Machine Learning Visualization Platform for Education - 15 algorithms, 9 datasets, AI tutor, code sandbox"
LABEL org.opencontainers.image.source="https://github.com/jakejrc/ML-Lab"

WORKDIR /app

# 安装中文字体（matplotlib 中文渲染需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

ENV MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
=======
FROM python:3.11-slim

LABEL maintainer="jake_jrc@qq.com"
LABEL org.opencontainers.image.title="ML-Lab"
LABEL org.opencontainers.image.description="Interactive Machine Learning Visualization Platform for Education - 15 algorithms, 9 datasets, AI tutor, code sandbox"
LABEL org.opencontainers.image.source="https://github.com/jakejrc/ML-Lab"

WORKDIR /app

# 安装中文字体（matplotlib 中文渲染需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

ENV MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
>>>>>>> Stashed changes
