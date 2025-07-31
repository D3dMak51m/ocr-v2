import pika
import json

# Setup connection and credentials
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='rabbitmq',
        port=5672,
        credentials=pika.PlainCredentials('admin', 'admin')
    )
)
channel = connection.channel()

# MUST match how queue was declared before (durable=True)
channel.queue_declare(queue='ocr_results', durable=True)

print("[📥] Reading messages from 'ocr_results' queue...\n")

while True:
    method_frame, header_frame, body = channel.basic_get(queue='ocr_results', auto_ack=False)
    if method_frame:
        try:
            print(" [✔] Got:", json.loads(body))
        except Exception:
            print(" [✔] Got (raw):", body.decode())

        # ✅ Manually acknowledge the message
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
    else:
        print("\n[✓] Queue is empty.")
        break

connection.close()
