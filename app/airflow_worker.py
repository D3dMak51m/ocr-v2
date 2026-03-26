import json
import logging
import os
import time

import pika
import pika.exceptions
from dotenv import load_dotenv

load_dotenv()


host = os.getenv("RABBITMQ_HOST", "rabbitmq")
credentials = pika.PlainCredentials(
    os.getenv("RABBITMQ_USER", "admin"),
    os.getenv("RABBITMQ_PASS", "admin")
)
params = pika.ConnectionParameters(host, credentials=credentials)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


def callback(ch, method, properties, body):
    try:
        data = json.loads(body)
        logger.info("Received message: %s", data)
        # TODO: add processing logic here.
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON message: %s", exc)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except Exception:
        logger.exception("Failed to process message")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)




max_retries = 10

for attempt in range(max_retries):
    try:
        connection = pika.BlockingConnection(params)
        logger.info("Connected to RabbitMQ")
        break
    except pika.exceptions.AMQPConnectionError:
        logger.warning("RabbitMQ not ready yet, retrying (%s/%s)...", attempt + 1, max_retries)
        time.sleep(2)
else:
    raise RuntimeError("Failed to connect to RabbitMQ after several retries")


channel = connection.channel()
channel.queue_declare(queue='ocr_results', durable=True)

# Start consuming
channel.basic_consume(queue='ocr_results', on_message_callback=callback)
logger.info("Waiting for messages in 'ocr_results'. To exit press CTRL+C")
channel.start_consuming()
