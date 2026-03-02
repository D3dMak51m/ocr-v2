FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Установка Python 3.11 и системных библиотек
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        python3.11 \
        python3.11-venv \
        python3-pip \
        python3.11-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Обновляем pip и базовые утилиты
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# =========================================================================
# КЭШИРУЕМЫЕ СЛОИ С ТЯЖЕЛЫМИ ML-ФРЕЙМВОРКАМИ
# =========================================================================

# 1. Устанавливаем PyTorch (нужен для YOLO/Ultralytics) строго для CUDA 11.8
# Это предотвратит долгую сборку из исходников и сэкономит место
RUN python3 -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# 2. Устанавливаем YOLO (Ultralytics). Теперь он не будет тянуть PyTorch заново.
RUN python3 -m pip install --no-cache-dir ultralytics==8.1.0

# 3. Устанавливаем PaddlePaddle GPU
RUN python3 -m pip install --no-cache-dir paddlepaddle-gpu==2.5.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# =========================================================================
# ЛЕГКИЕ ЗАВИСИМОСТИ ПРОЕКТА
# =========================================================================

COPY app/requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY app/ .

RUN mkdir -p /app/temp_files /app/models

RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so && \
    ln -sf /usr/local/cuda/lib64/libcublas.so.11 /usr/local/cuda/lib64/libcublas.so || true

EXPOSE 8282

ENV FLAGS_allocator_strategy=auto_growth
ENV OMP_THREAD_LIMIT=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8282", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--access-log", \
     "--limit-concurrency", "100"]