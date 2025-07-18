from airflow import DAG  # type: ignore
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator  # type: ignore
import requests
import json
import os
import logging


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
}


def call_fastapi_inference(**context):
    print("🟢 Entered call_fastapi_inference function")
    """Call FastAPI inference endpoint with all parameters passed from trigger"""
    # Get configuration from environment or use defaults
    fastapi_url = os.getenv("FAST_API_BASE_URL", "http://backend:8282")
    api_token = os.getenv("API_TOKEN", "asdjkhj8hsd!s8adhASas")

    # Get all parameters passed when DAG was triggered
    dag_run_conf = context["dag_run"].conf or {}

    # print(dag_run_conf)

    # Prepare OCR request with all parameters from trigger
    ocr_request = {
        "url": dag_run_conf.get(
            "url",
            "https://cf2.ppt-online.org/files2/slide/s/sEJXuRQk0xK4tH3ilIL1AMTB87dOmwcybo6aFSfpN/slide-0.jpg",
        ),
        "local_path": dag_run_conf.get("local_path", ""),
        "request_id": dag_run_conf.get("request_id", f"airflow_{context['ts_nodash']}"),
        "file_size_mb": dag_run_conf.get("file_size_mb", 1.0),
    }

    callback_url = dag_run_conf.get("callback_url", None)

    try:
        response = requests.post(
            f"{fastapi_url}/inference",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}",
            },
            json=ocr_request,
            timeout=300,  # 5 minutes timeout
        )

        if response.status_code < 300:
            result = response.json()
            # Store both the request and response for callback
            # context['task_instance'].xcom_push(key='inference_request', value=ocr_request)
            # context['task_instance'].xcom_push(key='inference_result', value=result)
            # return result
            requests.post(callback_url, json=result)
        else:
            error_response = {
                "status_code": response.status_code,
                "error": response.text,
                "request": ocr_request,
            }
            # context['task_instance'].xcom_push(key='inference_error', value=error_response)
            # raise Exception(f"FastAPI returned status {response.status_code}: {response.text}")
            requests.post(callback_url, json=error_response)
            return f"FastAPI returned status {response.status_code}: {response.text}"

    except Exception as e:
        error_response = {"exception": str(e), "request": ocr_request}
        # context['task_instance'].xcom_push(key='inference_error', value=error_response)
        # raise e
        requests.post(callback_url, json=error_response)
        return "Failed to call FastAPI inference"


# Remove the 'with' statement and assign DAG directly
dag = DAG(
    "airflow_large",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=False,
)

# Task 1: Call FastAPI inference with all trigger parameters
inference_task = PythonOperator(
    task_id="call_fastapi_inference",
    python_callable=call_fastapi_inference,
    dag=dag,
)
