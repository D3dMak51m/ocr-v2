FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir ultralytics==8.1.0

RUN pip install --no-cache-dir paddlepaddle-gpu==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

RUN mkdir -p /app/temp_files /app/models

EXPOSE 8282

ENV FLAGS_allocator_strategy=auto_growth
ENV OMP_THREAD_LIMIT=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8282", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--access-log", \
     "--limit-concurrency", "100"]