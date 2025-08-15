FROM python:3.11

USER root

# Install dependencies with optimization flags
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-rus \
        libtesseract-dev \
        libleptonica-dev \
        poppler-utils \
        libgomp1 \
        wget \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# RUN wget https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/rus.traineddata

COPY app/requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1

COPY app/ .

COPY ./app/traineddata/rus.traineddata /usr/share/tesseract-ocr/5/tessdata/
COPY ./app/traineddata/uzb_cyrl.traineddata /usr/share/tesseract-ocr/5/tessdata/
COPY ./app/traineddata/uzb.traineddata /usr/share/tesseract-ocr/5/tessdata/
COPY ./app/traineddata/en.traineddata /usr/share/tesseract-ocr/5/tessdata/


RUN mkdir -p /app/temp_files

ENV OMP_THREAD_LIMIT=4
ENV OMP_NUM_THREADS=4

ENV OPENCV_OPENCL_RUNTIME=disabled
ENV NUMEXPR_MAX_THREADS=4

# Python optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Tesseract optimizations
ENV TESSERACT_OPENCL_DEVICE=-1
ENV TESSERACT_DEBUG_LEVEL=0


EXPOSE 80
# --root-path /api/ --forwarded-allow-ips "*"
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8282"]

# Run with optimized worker settings
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8282", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--access-log", \
     "--limit-concurrency", "100"]