# 📄 OCR Service
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green?logo=fastapi)](https://fastapi.tiangolo.com/)  
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)  
[![Tesseract OCR](https://img.shields.io/badge/Tesseract-OCR-orange?logo=google)](https://github.com/tesseract-ocr/tesseract)  


An OCR (Optical Character Recognition) service that extracts text from images and from images embedded in documents (DOCX, PPTX, PDF, etc.), and detects the language of the extracted text.
The service uses:
- **FastAPI** for the REST API
- **Tesseract OCR** for text extraction
- **Custom Language Detection API** for language identification

---

## 🚀 Getting Started

### 1️⃣ Create `.env` file
Create a `.env` file in the root directory.  
You can use [`example_env.txt`](example_env.txt) as a reference.

---

### 2️⃣ Start the Services
From the project root, run:
```bash
docker compose up --build -d
```

---

## 🌐 Available Services

| Service           | URL                                         | Credentials       |
|-------------------|---------------------------------------------|-------------------|
| **Backend Docs**  | [http://localhost:8282/docs](http://localhost:8282/docs) | Token required   |
| **Airflow UI**    | [http://localhost:8080](http://localhost:8080)           | admin / admin   |
| **RabbitMQ UI**   | [http://localhost:15672](http://localhost:15672)         | admin / admin   |

> 📝 Airflow credentials can be changed in the `.env` file.

---

## 📨 RabbitMQ Consumer-Client Samples
RabbitMQ consumers are available in [`client_sample`](doc/client_sample) directory:
- [`consume.py`](doc/client_sample/consume.py)  
- [`consume2.py`](doc/client_sample/consume2.py) (if the first one doesn't work)

---
## 📤 Example API Requests

### 1. Direct Inference (Synchronous)
```bash
curl --location 'http://localhost:8282/ocr/inference' --header 'Content-Type: application/json' --header 'Authorization: Bearer YOUR_TOKEN' --data '{
  "url": "https://cf2.ppt-online.org/files2/slide/s/sEJXuRQk0xK4tH3ilIL1AMTB87dOmwcybo6aFSfpN/slide-0.jpg"
}'
```
* The maximum allowed file size is 50 MB
  
---

### 2. Queue Inference (Asynchronous)
```bash
curl --location 'http://localhost:8282/ocr/queue_inference' --header 'Content-Type: application/json' --header 'Authorization: Bearer YOUR_TOKEN' --data '{
   "url": "https://yourpositiveoasis.com/wp-content/gallery/25-inspiring-and-positive-quotes/IMG_7101.PNG",
   "local_path": "",
   "request_id": "string_5",
   "file_size_mb": 3,
   "callback_url": "https://webhook-test.com/ba547e1d07c3133881c8a516ca338cb1"
}'
```
➡ The response will be `"received"`. The result can be retrieved by listening to RabbitMQ on channel `ocr_results`.

---

## 📑 Supported File Formats
- `jpeg`
- `jpg`
- `png`
- `gif`
- `bmp`
- `pdf`

---

## 📌 Architecture Overview
```mermaid
flowchart TD
    A[Client Request] -->|API Call| B[FastAPI OCR Service]
    B --> C[Tesseract OCR]
    B --> D[Language Detection API]
    B --> E[RabbitMQ Queue / ocr_results]
    F[Client - Consumer] --> E
    
```

---
## 🛠 Docker Compose Setup

## 🏗 Architecture Overview

The OCR Service is composed of several connected components that work together for text extraction, document parsing, and workflow orchestration.

```mermaid
  flowchart LR
    %% Client Service
    subgraph ClientService["Client Service"]
        CB_Service["Client Service"]
    end

    %% Backend Service
    subgraph Backend["Backend - FastAPI OCR Service"]
        BE_API["API - Uvicorn"]
        BE_Tess["Tesseract OCR - Local"]
        BE_Tika["Tika API Client"]
    end
    
    %% Tika Server
    subgraph Tika["Tika-server"]
        TIKA_APP["Tika OCR / Document Parser"]
    end

    %% Airflow Service
    subgraph Airflow["Airflow"]
        AF_Web["Webserver - REST API"]
        AF_DAG["Airflow DAGs - airflow_dag, large_dag"]
        AF_Tool["tool.py"]
        AF_Rabbit["Queue Producer"]
    end
    
    %% RabbitMQ
    subgraph MQ["RabbitMQ"]
        MQ_QUEUE["ocr_results Queue"]
    end

    %% Postgres DB
    subgraph DB["Postgres"]
        PG_DB["Airflow Metastore DB"]
    end

    %% Connections
    CB_Service --> BE_API
    BE_API --> BE_Tess
    BE_API --> BE_Tika
    BE_Tika --> TIKA_APP

    %% Backend triggers Airflow Webserver
    BE_API -->|queue_inference| AF_Web
    
    AF_Web --> AF_DAG
    AF_DAG -->|LargeDag - Call | AF_Tool
    AF_Tool -->|Trigger| BE_API
    AF_Tool -->|Publish Result by producer| AF_Rabbit
    AF_Rabbit --> |Produce| MQ_QUEUE
    MQ_QUEUE -->|"Consume (Result of large files)"| CB_Service

    Airflow --> PG_DB
```

---

## 🛠 Tech Stack
- **Backend:** FastAPI
- **OCR Engine:** Tesseract
- **Queue:** RabbitMQ
- **Workflow Orchestration:** Airflow
- **Containerization:** Docker

---