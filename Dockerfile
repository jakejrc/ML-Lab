FROM python:3.11-slim AS builder

WORKDIR /app

# 仅复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

LABEL maintainer="jake_jrc@qq.com"
LABEL org.opencontainers.image.title="ML-Lab"
LABEL org.opencontainers.image.description="ML-Lab v3.4 - Interactive ML Visualization Platform | 15 algorithms, 9 datasets, tabular evaluation, AI tutor, code sandbox"
LABEL org.opencontainers.image.source="https://github.com/jakejrc/ML-Lab"
LABEL org.opencontainers.image.version="3.4"

WORKDIR /app

# 安装中文字体（matplotlib 中文渲染需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 从 TTC 提取 SC 子字体为独立 OTF（matplotlib 3.10+ 无法直接从 TTC 加载 SC 变体）
RUN pip install --no-cache-dir fonttools && \
    python3 -c " \
from fontTools.ttLib import TTCollection; \
ttc = TTCollection('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'); \
ttc.fonts[2].save('/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf'); \
ttc2 = TTCollection('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'); \
ttc2.fonts[2].save('/usr/share/fonts/opentype/noto/NotoSansCJKSC-Bold.otf')" \
    && pip uninstall -y fonttools && \
    fc-cache -fv

# 从 builder 复制已安装的 Python 包
COPY --from=builder /install /usr/local

ENV MPLBACKEND=Agg

COPY . .

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
