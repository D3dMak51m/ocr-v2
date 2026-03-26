import numpy as np
import cv2
import paddle
import paddle.nn as nn


class UzbekRecognizer:
    def __init__(self, model_dir, char_dict_path, max_text_length=60):
        paddle.set_device('cpu')
        self.model = paddle.jit.load(f'{model_dir}/inference')
        self.model.eval()

        with open(char_dict_path, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f]
        self.character = ['blank'] + chars
        self.max_text_length = max_text_length

    def preprocess(self, img):
        h, w = img.shape[:2]
        target_h = 48
        target_w = int(w * target_h / h)
        target_w = max(target_w, 1)
        img = cv2.resize(img, (target_w, target_h))
        # Pad short images to minimum width of 320
        if target_w < 320:
            pad = np.zeros((target_h, 320 - target_w, 3), dtype=np.float32)
            img = np.concatenate([img, pad], axis=1)
        # If wider than 320, keep the actual width — model supports dynamic input
        img = img.astype('float32') / 255.0
        img -= 0.5
        img /= 0.5
        img = img.transpose((2, 0, 1))
        return img[np.newaxis, :]

    def postprocess(self, preds):
        preds = np.argmax(preds, axis=2)[0]
        result = []
        prev = -1
        for idx in preds:
            if idx != prev and idx != 0:
                if idx < len(self.character):
                    result.append(self.character[idx])
            prev = idx
        return ''.join(result)

    def predict(self, img):
        data = self.preprocess(img)
        tensor = paddle.to_tensor(data)
        with paddle.no_grad():
            output = self.model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        return self.postprocess(output.numpy())