import pika
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()


host = os.getenv("RABBITMQ_HOST", "rabbitmq")
credentials = pika.PlainCredentials(
    os.getenv("RABBITMQ_USER", "admin"),
    os.getenv("RABBITMQ_PASS", "admin")
)
params = pika.ConnectionParameters(host, credentials=credentials)

print(os.getenv("RABBITMQ_USER", "admin"))
print(os.getenv("RABBITMQ_PASS", "admin"))


def callback(ch, method, properties, body):
    data = json.loads(body)
    print(f"Received message: {data}")




max_retries = 10

for attempt in range(max_retries):
    try:
        connection = pika.BlockingConnection(params)
        print("Connected to RabbitMQ")
        break
    except pika.exceptions.AMQPConnectionError as e:
        print(f"RabbitMQ not ready yet, retrying ({attempt + 1}/{max_retries})...")
        time.sleep(2)
else:
    raise RuntimeError("Failed to connect to RabbitMQ after several retries")


channel = connection.channel()
channel.queue_declare(queue='ocr_results', durable=True)

# Start consuming
channel.basic_consume(queue='ocr_results', on_message_callback=callback)
print(f"🔁 Waiting for messages in 'ocr_results'. To exit press CTRL+C")
channel.start_consuming()