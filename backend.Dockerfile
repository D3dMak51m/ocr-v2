FROM python:3.11

USER root

RUN apt-get update && \
    apt-get install -y tesseract-ocr poppler-utils wget && \
    rm -rf /var/lib/apt/lists/*


RUN apt install wget
WORKDIR /app

# RUN wget https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/rus.traineddata

COPY app/requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ .

COPY ./app/rus.traineddata /usr/share/tesseract-ocr/5/tessdata/
COPY ./app/uzb_cyrl.traineddata /usr/share/tesseract-ocr/5/tessdata/
COPY ./app/uzb.traineddata /usr/share/tesseract-ocr/5/tessdata/
COPY ./app/en.traineddata /usr/share/tesseract-ocr/5/tessdata/


RUN mkdir -p /app/temp_files

EXPOSE 80
# --root-path /api/ --forwarded-allow-ips "*"
CMD ["uvicorn", "main:webapp", "--host", "0.0.0.0", "--port", "8282"]

