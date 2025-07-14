# OCR Service README

This project provides an OCR (Optical Character Recognition) service to extract text from images and detect the language of the extracted text. The service uses FastAPI for the API interface, Tesseract OCR for text extraction, and a custom language detection API for determining the language of the text.

# Run instructions:


## 1. Create env file:

create .env file in the root folder and add (change password and tokens accordingly):

```sh
# .env

# API Token
API_TOKEN=asdjkhj8hsd!s8adhASas


# Postgres info for airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

## 2. Run docker compose

From the root folder, simply run:

```sh
docker compose up --build -d
```

# Example Requests:

## 1. Direct response inference:

```sh

curl --location 'http://0.0.0.0:8282/ocr/inference' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer asdjkhj8hsd!s8adhASas' \
--data '{
  "url": "https://cf2.ppt-online.org/files2/slide/s/sEJXuRQk0xK4tH3ilIL1AMTB87dOmwcybo6aFSfpN/slide-0.jpg"
}'

```

## 2. Sending request into queue:

```sh

curl --location 'http://0.0.0.0:8282/ocr/queue_inference' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer asdjkhj8hsd!s8adhASas' \
--data '{
   "url": "https://yourpositiveoasis.com/wp-content/gallery/25-inspiring-and-positive-quotes/IMG_7101.PNG",
   "local_path": "",
   "request_id": "string_5",
   "file_size_mb": 3,
   "callback_url": "https://webhook-test.com/ba547e1d07c3133881c8a516ca338cb1"
}'

```

You will get a received response, and to check the result listen on the rabbitMQ. An example of listening to rabbitMQ is in the `client_sample` folder.


### 🌐 Accessing Airflow Web UI

After starting the services with Docker Compose, you can access the Airflow web interface at:

- 📍 **URL:** [http://localhost:8080](http://localhost:8080)

#### 🛠 Port Mapping

The Airflow webserver is exposed on:

password, login admin admin

```bash
ports:
  - "8080:8080"  # host:container

 
## Supported extensions:

- "jpeg/jpg",
- "jpeg",
- "jpg",
- "png",
- "gif",
- "bmp",
- "pdf"