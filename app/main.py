from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app import runner
import os
from ocr_models.bmodels import TextractRequest, AirflowTask
import requests
from fastapi.middleware.cors import CORSMiddleware

from worker import process_ocr_task_small, process_ocr_task_large

# Read root_path from environment variable, default to "/ocr"
root_path = "/ocr"

webapp = FastAPI()


origins = ["*"]

webapp.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

token_auth_scheme = HTTPBearer()

API_TOKEN = os.getenv("API_TOKEN", "your_secret_token")


def verify_token(http_auth: HTTPAuthorizationCredentials = Depends(token_auth_scheme)):
    if http_auth.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token"
        )


@webapp.get("/")
async def read_root():
    return {"message": "Hello from FastAPI, OCR service is running!"}


@webapp.post("/inference")
def text_extraction(
    request: TextractRequest,
    token: HTTPAuthorizationCredentials = Depends(verify_token),
) -> dict:
    """
    ## Supported extensions:

    - "jpeg/jpg",
    - "jpeg",
    - "jpg",
    - "png",
    - "gif",
    - "bmp",
    - "pdf"
    - "doc",
    - "docx",
    - "ppt",
    - "pptx"
    """
    try:
        if request.file_size_mb > 50:
            # can not handle files larger than 50MB
            return {
                "request_id": request.request_id,
                "status": "File size is too large. Please use the \
                    /queue_inference endpoint for files larger than 50MB.",
            }
        result = runner(request)
        resp = {"request_id": request.request_id, "result": result}
        return resp
    except Exception as e:
        error_msg = str(e)
        print(Exception, e)
        return {"status": error_msg}


@webapp.post("/airflow_task")
def create_airflow_task(
    request: AirflowTask,
    token: HTTPAuthorizationCredentials = Depends(verify_token),
) -> dict:

    AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://airflow-webserver:8080")
    # print(AIRFLOW_BASE_URL)
    airflow_url = f"{AIRFLOW_BASE_URL}/api/v1/dags/airflow_dag/dagRuns"
    airflow_url_large = f"{AIRFLOW_BASE_URL}/api/v1/airflow_large/dagRuns"

    if request.file_size_mb <= 5:
        preffered_dag_url = airflow_url
    else:
        preffered_dag_url = airflow_url_large

    response = requests.post(
        preffered_dag_url, json={"conf": request.dict()}, auth=("admin", "admin")
    )
    print(f"Response from Airflow: {response.status_code}, {response.text}")

    if response.status_code == 200:
        return {"request_id": request.request_id, "status": "received"}
    else:
        return {"request_id": request.request_id, "status": "failed"}
