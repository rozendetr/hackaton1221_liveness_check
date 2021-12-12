import numpy as np

from .tools import *
from .detector import *
from .classificator import *
import uuid
import os



class NNPipeLine:
    def __init__(self,
                 logger=None,
                 detector: Detector = None,
                 spoof_classificator = None):
        self.logger = logger
        self.detector = detector
        self.spoof_classificator = spoof_classificator
        self.setting = {"detect_face_threshold": 0.3,
                        "min_face_size": 5}

    def predict_images(self, batch_imgs: list):
        batch_bboxes = self.detector(batch_imgs,
                                     detector_threshold=self.setting.get("detect_face_threshold", 0.3),
                                     min_size_object=self.setting.get("min_face_size", 5))
        if not batch_bboxes:
            return None

        batch_detections = []
        for id_batch, bboxes in enumerate(batch_bboxes):
            frame = batch_imgs[id_batch]
            spoof_bboxes = []
            for bbox in bboxes:
                x_tl, y_tl, x_br, y_br = bbox
                w_bbox = x_br-x_tl
                h_bbox = y_br-y_tl
                frame_crop = frame[y_tl:y_br, x_tl:x_br]
                class_id = self.spoof_classificator(frame_crop)
                if class_id in [7]:
                    class_id = 0
                spoof_bbox = [x_tl, y_tl, x_br, y_br, class_id]
                spoof_bboxes.append(spoof_bbox)
            batch_detections.append(spoof_bboxes)

        return batch_detections



