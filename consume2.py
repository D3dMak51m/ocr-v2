import pika
import sys

# Connect to RabbitMQ
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host="localhost", port=5672, credentials=pika.PlainCredentials("admin", "admin")
    )
)
channel = connection.channel()

queue_name = "ocr_results"

# Declare the queue
channel.queue_declare(queue=queue_name, durable=True)

# Cancel any existing consumers (this might help release unacked messages)
try:
    channel.cancel()
except:
    pass

# Close and reopen connection to force release of unacked messages
connection.close()

# Reconnect
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host="localhost", port=5672, credentials=pika.PlainCredentials("admin", "admin")
    )
)
channel = connection.channel()

# Now try to consume
channel.queue_declare(queue=queue_name, durable=True)

print("[🔄] Attempting to recover and consume messages...")

count = 0
while True:
    method_frame, header_frame, body = channel.basic_get(
        queue=queue_name, auto_ack=True
    )
    if method_frame:
        print(f" [✔] Message {count + 1}:", body.decode())
        count += 1
    else:
        break

print(f"\n[✓] Consumed {count} messages")
connection.close()
