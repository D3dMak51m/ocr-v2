# OCR Service README

This project provides an OCR (Optical Character Recognition) service to extract text from images and detect the language of the extracted text. The service uses FastAPI for the API interface, Tesseract OCR for text extraction, and a custom language detection API for determining the language of the text.

# Run instructions:


## 1. Create env file:

create .env file in the root folder and add (change password and tokens accordingly):

```sh
# .env

# RabbitMQ Configuration
RABBITMQ_USER=admin
RABBITMQ_PASSWORD=admin_ocr_123SjC7s
RABBITMQ_HOST=rabbitmq

# Celery Configuration
CELERY_BROKER_URL=amqp://admin:admin_ocr_123SjC7s@rabbitmq:5672//

# API Token
API_TOKEN=asdjkhj8hsd!s8adhASas

# Flower Configuration
FLOWER_USER=admin
FLOWER_PASSWORD=admin_ocr_123SjC7s

```

## 2. Run docker compose

From the root folder, simply run:

```sh
docker compose up -d
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
  "url": "https://cf2.ppt-online.org/files2/slide/s/sEJXuRQk0xK4tH3ilIL1AMTB87dOmwcybo6aFSfpN/slide-0.jpg"
}'

```

You will get a received response, and to check the result listen on the rabbitMQ. An example of listening to rabbitMQ is in the `client_sample` folder.

## Instructions to run the sample that listens on the RabbitMQ

### Prerequisites
Make sure you have the following installed:
1. **Python**: Run `python3 --version` to check if Python is installed. If not, install it using:
   ```bash
   sudo apt update
   sudo apt install python3
   ```
2. **pip**: Run `pip3 --version` to check if pip is installed. If not, install it using:
   ```bash
   sudo apt install python3-pip
   ```
3. **venv**: Ensure the `venv` package is installed:
   ```bash
   sudo apt install python3-venv
   ```

### Instructions

1. **Navigate to your project directory**:
   ```bash
   cd client_sample
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

4. **Install the required packages from `requirements.txt`**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run your Python script** (do not forget to check ip / port and login credentials of the RabbitMQ):
   ```bash
   python3 consumer.py
   ```

6. **Deactivate the virtual environment after you are done**:
   ```bash
   deactivate
   ```



## Supported extensions:

- "jpeg/jpg",
- "jpeg",
- "jpg",
- "png",
- "gif",
- "bmp",
- "pdf"