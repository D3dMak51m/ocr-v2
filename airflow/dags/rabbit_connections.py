import pika
import json

def add_queue(message: dict):
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host="rabbitmq",
                credentials=pika.PlainCredentials("admin", "admin"),
            )
        )
        channel = connection.channel()

        # 🔥 Key line: declare queue before publishing
        channel.queue_declare(queue="ocr_results", durable=True)

        body = json.dumps(message)
        print("📤 Publishing message:", body)

        # 🔥 Publish
        channel.basic_publish(
            exchange="",
            routing_key="ocr_results",
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )

        print("✅ Message published successfully!")
        connection.close()

    except Exception as e:
        print("❌ Failed to publish message:", e)
