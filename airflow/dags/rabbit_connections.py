"""
RabbitMQ connection utilities
"""

import pika
import json
import logging
import os
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Configuration
RABBITMQ_CONFIG = {
    "host": os.getenv("RABBITMQ_HOST", "rabbitmq"),
    "port": int(os.getenv("RABBITMQ_PORT", "5672")),
    "username": os.getenv("RABBITMQ_DEFAULT_USER", "admindefault"),
    "password": os.getenv("RABBITMQ_DEFAULT_PASS", "admin_ocr_123SjC7s"),
    "queue_name": os.getenv("RABBITMQ_QUEUE", "ocr_results"),
    "connection_timeout": int(os.getenv("RABBITMQ_TIMEOUT", "10")),
}


@contextmanager
def get_rabbitmq_connection():
    """
    Context manager for RabbitMQ connections
    Ensures proper connection cleanup
    """
    connection = None
    try:
        # Create connection with timeout
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBITMQ_CONFIG["host"],
                port=RABBITMQ_CONFIG["port"],
                credentials=pika.PlainCredentials(
                    RABBITMQ_CONFIG["username"], RABBITMQ_CONFIG["password"]
                ),
                connection_attempts=3,
                retry_delay=1,
                socket_timeout=RABBITMQ_CONFIG["connection_timeout"],
            )
        )
        yield connection
    except pika.exceptions.AMQPConnectionError as e:
        logger.error(f"Failed to connect to RabbitMQ: {e}")
        raise
    finally:
        if connection and not connection.is_closed:
            connection.close()


def add_response_to_queue(message: Dict[str, Any]) -> bool:
    """
    Add message to RabbitMQ queue

    Args:
        message: Dictionary containing the message to be sent

    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    try:
        with get_rabbitmq_connection() as connection:
            channel = connection.channel()

            # Declare queue (idempotent operation)
            channel.queue_declare(queue=RABBITMQ_CONFIG["queue_name"],
                                   durable=True)

            # Serialize message
            body = json.dumps(message, ensure_ascii=False)

            # Log message details
            logger.info(
                f"📤 Publishing message to queue '{RABBITMQ_CONFIG['queue_name']}': "
                f"request_id={message.get('request_id', 'N/A')}"
            )

            # Publish message with persistence
            channel.basic_publish(
                exchange="",
                routing_key=RABBITMQ_CONFIG["queue_name"],
                body=body.encode("utf-8"),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    content_type="application/json",
                ),
                mandatory=True,  # Ensure message is routed to a queue
            )

            # Confirm delivery
            if channel.is_open:
                logger.info("✅ Message published successfully!")
                return True
            else:
                logger.error("❌ Channel closed unexpectedly")
                return False

    except pika.exceptions.UnroutableError:
        logger.error(
            f"❌ Message could not be routed to queue '{RABBITMQ_CONFIG['queue_name']}'"
        )
        return False
    except Exception as e:
        logger.error(f"❌ Failed to publish message: {type(e).__name__}: {e}")
        return False


def get_queue_info() -> Optional[Dict[str, Any]]:
    """
    Get information about the queue (message count, consumer count, etc.)

    Returns:
        Dictionary with queue information or None if error
    """
    try:
        with get_rabbitmq_connection() as connection:
            channel = connection.channel()

            # Declare queue to ensure it exists
            method = channel.queue_declare(
                queue=RABBITMQ_CONFIG["queue_name"],
                durable=True,
                passive=True,  # Don't create, just check
            )

            return {
                "queue_name": RABBITMQ_CONFIG["queue_name"],
                "message_count": method.method.message_count,
                "consumer_count": method.method.consumer_count,
            }

    except Exception as e:
        logger.error(f"Failed to get queue info: {e}")
        return None
