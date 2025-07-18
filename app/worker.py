from celery import Celery
import os
from app import runner
from kombu import Connection, Exchange, Producer, Queue
from ocr_models.bmodels import TextractRequest
from fastapi.encoders import jsonable_encoder

RABBITMQ_USER = os.getenv("RABBITMQ_USER", "admin")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "admin_ocr_123SjC7s")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")


broker_url = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:5672//"

celery_app = Celery("worker", broker=broker_url)

# Define the queues
celery_app.conf.task_queues = (
    Queue("small_files", routing_key="small_files", durable=True),
    Queue("large_files", routing_key="large_files", durable=True),
)

celery_app.conf.task_routes = {
    "process_small_file": {"queue": "small_files"},
    "process_large_file": {"queue": "large_files"},
}


def handler(url: str, local_path: str, request_id: str, file_size_mb: float):
    request = TextractRequest(
        url=url, local_path=local_path, request_id=request_id, file_size_mb=file_size_mb
    )
    result = runner(request)
    with Connection(broker_url) as conn:
        exchange = Exchange("results", type="direct", durable=True)
        queue = Queue(
            name="results_queue",
            exchange=exchange,
            routing_key="ocr_results",
            durable=True,
            delivery_mode=Exchange.PERSISTENT_DELIVERY_MODE,
        )
        queue(conn).declare()  # Declare the queue

        producer = Producer(conn)
        message = {"request_id": request.request_id, "result": jsonable_encoder(result)}
        producer.publish(
            message,
            exchange=exchange,
            routing_key="ocr_results",
            serializer="json",
            durable=True,
            delivery_mode=Exchange.PERSISTENT_DELIVERY_MODE,
        )


@celery_app.task(name="process_small_file")
def process_ocr_task_small(url, local_path, request_id, file_size_mb):
    # Process files smaller than 50MB
    return handler(url, local_path, request_id, file_size_mb)


@celery_app.task(name="process_large_file")
def process_ocr_task_large(url, local_path, request_id, file_size_mb):
    # Process files larger than 50MB
    return handler(url, local_path, request_id, file_size_mb)
