"""
Common utility functions for Airflow DAGs
"""

import os
import logging
import requests
import json
from typing import Dict, Any
from rabbit_connections import add_response_to_queue

logger = logging.getLogger(__name__)


def get_config() -> Dict[str, str]:
    """Get configuration from environment variables"""
    return {
        "fastapi_url": os.getenv("FAST_API_BASE_URL", "http://backend:8282"),
        "api_token": os.getenv("API_TOKEN", "asdjkhj8hsd!s8adhASas"),
    }


def prepare_ocr_request(
    dag_run_conf: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare OCR request payload from DAG run configuration"""
    return {
        "url": dag_run_conf.get(
            "url",
            "https://cf2.ppt-online.org/files2/slide/s/sEJXuRQk0xK4tH3ilIL1AMTB87dOmwcybo6aFSfpN/slide-0.jpg",
        ),
        "local_path": dag_run_conf.get("local_path", ""),
        "request_id": dag_run_conf.get("request_id", f"airflow_{context['ts_nodash']}"),
        "file_size_mb": dag_run_conf.get("file_size_mb", 1.0),
    }


def call_fastapi_inference(**context) -> None:
    """
    Call FastAPI inference endpoint with all parameters passed from trigger
    Sends successful results to RabbitMQ queue
    Raises exceptions on errors to fail the Airflow task
    """
    logger.info("🟢 Starting FastAPI inference call")
    logger.info(f"Task Instance: {context['task_instance'].task_id}")
    logger.info(f"DAG Run ID: {context['dag_run'].run_id}")

    # Get configuration
    config = get_config()
    fast_api_inference_url = config["fastapi_url"] + "/api/v2/inference"

    # Get parameters from DAG trigger
    dag_run_conf = context["dag_run"].conf or {}

    # Prepare request
    ocr_request = prepare_ocr_request(dag_run_conf, context)

    # Log the request details
    logger.info("📤 OCR Request Details:")
    logger.info(f"URL: {ocr_request.get('url', 'N/A')}")
    logger.info(f"Request ID: {ocr_request.get('request_id', 'N/A')}")
    logger.info(f"Local Path: {ocr_request.get('local_path', 'N/A')}")
    logger.info(f"File Size: {ocr_request.get('file_size_mb', 'N/A')} MB")

    try:
        # log fastapi url with inference and token
        logger.info(
            f"🔗 Calling FastAPI inference endpoint: {fast_api_inference_url}"
        )
        logger.info(f"🔑 Using API Token: {config['api_token']}")
        # Call FastAPI endpoint
        response = requests.post(
            fast_api_inference_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['api_token']}",
            },
            json=ocr_request,
            timeout=300,  # 5 minutes timeout
        )

        if response.status_code < 300:
            # Success - send to RabbitMQ
            result = response.json()
            logger.info(
                f"✅ Inference successful for request_id: {ocr_request['request_id']}"
            )

            # Send to RabbitMQ queue
            if not add_response_to_queue(result):
                # If RabbitMQ send fails, raise exception to fail the task
                raise Exception("Failed to send result to RabbitMQ queue")
            # requests.post("https://webhook-test.com/449db4f6a63b66850a427b0af1f3da3f", 
            #               json=result)
            # Log success summary
            logger.info("=" * 80)
            logger.info(f"✅ OCR TASK COMPLETED SUCCESSFULLY")
            logger.info(f"   Request ID: {ocr_request['request_id']}")
            logger.info(f"   URL: {ocr_request['url']}")

            # Log the OCR results for visibility in Airflow UI
            logger.info("📄 OCR Results:")
            logger.info(f"Request ID: {result.get('request_id', 'N/A')}")
            logger.info(f"Status: {result.get('status', 'N/A')}")

            logger.info(f"!!!Result: {result}")

            logger.info("Full OCR Response:")
            logger.info(json.dumps(result, indent=2, ensure_ascii=False))

            return

        else:
            # API error - raise exception to fail the Airflow task
            error_msg = (
                f"FastAPI returned status {response.status_code}: {response.text}"
            )
            logger.error(f"❌ {error_msg}")

            # Optionally store error details in XCom for debugging
            context["task_instance"].xcom_push(
                key="error_details",
                value={
                    "status_code": response.status_code,
                    "error": response.text,
                    "request": ocr_request,
                },
            )

            # Raise exception to fail the task
            raise requests.HTTPError(error_msg)

    except requests.Timeout as e:
        # Timeout error
        error_msg = f"Request timeout after 300 seconds: {str(e)}"
        logger.error(f"❌ {error_msg}")

        # Store error in XCom
        context["task_instance"].xcom_push(
            key="error_details",
            value={
                "exception": "Timeout",
                "message": str(e),
                "request": ocr_request,
            },
        )

        # Re-raise to fail the task
        raise

    except requests.RequestException as e:
        # Network/Request error
        error_msg = f"Request failed: {str(e)}"
        logger.error(f"❌ {error_msg}")

        # Store error in XCom
        context["task_instance"].xcom_push(
            key="error_details",
            value={
                "exception": type(e).__name__,
                "message": str(e),
                "request": ocr_request,
            },
        )

        # Re-raise to fail the task
        raise

    except Exception as e:
        # Any other exception
        error_msg = f"Unexpected error during inference: {str(e)}"
        logger.exception(f"❌ {error_msg}")

        # Store error in XCom
        context["task_instance"].xcom_push(
            key="error_details",
            value={
                "exception": type(e).__name__,
                "message": str(e),
                "request": ocr_request,
            },
        )

        # Re-raise to fail the task
        raise
