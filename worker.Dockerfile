FROM python:3.11

USER root

RUN apt-get update && \
    apt-get install -y tesseract-ocr poppler-utils wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

COPY ./app/rus.traineddata /usr/share/tesseract-ocr/5/tessdata/
COPY ./app/uzb_cyrl.traineddata /usr/share/tesseract-ocr/5/tessdata/
COPY ./app/uzb.traineddata /usr/share/tesseract-ocr/5/tessdata/
COPY ./app/en.traineddata /usr/share/tesseract-ocr/5/tessdata/


RUN mkdir -p /app/temp_files

ENV PYTHONPATH=/app

# CMD ["celery", "-A", "worker", "worker", "--loglevel=info", "-Q", "${CELERY_QUEUE}"]
