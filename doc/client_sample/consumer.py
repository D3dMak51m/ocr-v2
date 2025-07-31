import os
import pika
import json

RABBITMQ_USER = os.getenv("RABBITMQ_USER", "admin")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "admin_ocr_123SjC7s")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "139.28.47.17")

# Pika connection parameters
credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
parameters = pika.ConnectionParameters(RABBITMQ_HOST, 5556, "/", credentials)


# Callback function for processing messages
def process_message(ch, method, properties, body):
    print(f"Received message: {json.loads(body)}")
    ch.basic_ack(delivery_tag=method.delivery_tag)


# Establish connection
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

# Declare exchange and queue
channel.exchange_declare(exchange="results", exchange_type="direct", durable=True)
channel.queue_declare(queue="results_queue", durable=True)
channel.queue_bind(exchange="results", queue="results_queue", routing_key="ocr_results")

# Set up the consumer
channel.basic_consume(queue="results_queue", on_message_callback=process_message)

print("Waiting for messages...")
try:
    # Start consuming messages
    channel.start_consuming()
except KeyboardInterrupt:
    print("Interrupted, closing connection...")
    channel.stop_consuming()

# Close the connection
connection.close()
