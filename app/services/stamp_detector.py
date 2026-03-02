import logging
import os
from typing import List
from PIL import Image
from core.schemas import DetectedStamp, BoundingBox

logger = logging.getLogger(__name__)


class StampDetector:
    def __init__(self):
        self.model_path = os.getenv("STAMP_MODEL_PATH", "yolov8n.pt")
        self.model = None

    def detect(self, pil_image: Image.Image) -> List[DetectedStamp]:
        if self.model is None:
            logger.info("Initializing YOLO model in worker...")
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                return []

        stamps = []
        try:
            results = self.model.predict(source=pil_image, conf=0.5, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]

                    stamps.append(DetectedStamp(
                        label=label,
                        confidence=conf,
                        box=BoundingBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
                    ))
        except Exception as e:
            logger.error(f"YOLO prediction error: {e}")

        return stamps


stamp_detector = StampDetector()