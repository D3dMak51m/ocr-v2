import pika
import json


def callback(ch, method, properties, body):
    data = json.loads(body)
    print(f"Received message: {data}")


connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host='rabbitmq'
    )
)

channel = connection.channel()


channel.queue_declare(queue='ocr_results', durable=True)


channel.basic_consume(queue='ocr_results', on_message_callback=callback, auto_ack=True)

print('Waiting for messages. To exit press CTRL+C')
channel.start_consuming()