FROM python:3.11-slim

LABEL maintainer="jake_jrc@qq.com"
LABEL description="ML-Lab: 机器学习可视化实验平台"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]