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

# webapp = FastAPI(
#     root_path=root_path,
#     docs_url='/docs',
#     redoc_url='/redoc',
#     openapi_url='/openapi.json')

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


# @webapp.post("/queue_inference")
# def queue_text_extraction(
#     request: TextractRequest,
#     token: HTTPAuthorizationCredentials = Depends(verify_token),
# ) -> dict:
#     """
#     ## Supported extensions:

#     - "jpeg/jpg",
#     - "jpeg",
#     - "jpg",
#     - "png",
#     - "gif",
#     - "bmp",
#     - "pdf"
#     - "doc",
#     - "docx",
#     - "ppt",
#     - "pptx"
#     """

#     if request.file_size_mb <= 50:
#         process_ocr_task_small.delay(
#             request.url, request.local_path, request.request_id, request.file_size_mb
#         )
#     else:
#         process_ocr_task_large.delay(
#             request.url, request.local_path, request.request_id, request.file_size_mb
#         )
#     return {"request_id": request.request_id, "status": "received"}


@webapp.post("/airflow_task")
def create_airflow_task(
    request: AirflowTask,
    token: HTTPAuthorizationCredentials = Depends(verify_token),
) -> dict:

    AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")
    print(AIRFLOW_BASE_URL)
    airflow_url = f"{AIRFLOW_BASE_URL}/api/v1/dags/airflow_dag/dagRuns"
    response = requests.post(
        airflow_url,
        json={"conf": request.dict()}, 
        auth=("admin", "admin")
    )
    print(f"Response from Airflow: {response.status_code}, {response.text}")

    if response.status_code == 200:
        return {"request_id": request.request_id, "status": "received"}
    else:
        return {"request_id": request.request_id, "status": "failed"}