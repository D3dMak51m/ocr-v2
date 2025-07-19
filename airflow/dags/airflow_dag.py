"""
Airflow DAG for standard OCR inference processing
"""
from airflow import DAG  # type: ignore
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator  # type: ignore
from tool import call_fastapi_inference

# DAG default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
}

# Create DAG
dag = DAG(
    dag_id="airflow_dag",
    default_args=default_args,
    description="Standard OCR inference DAG",
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=["ocr", "inference", "standard"],
)

# Task: Call FastAPI inference endpoint
inference_task = PythonOperator(
    task_id="call_fastapi_inference",
    python_callable=call_fastapi_inference,
    dag=dag,
)