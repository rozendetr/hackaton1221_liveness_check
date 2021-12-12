import cv2

from .detector_base import Detector
import tensorflow.compat.v1 as tf
import numpy as np

from ..tools import init_pb


class SSD(Detector):
    def __init__(self, pb_path, input_res: tuple = (640, 640), detector_margin=0.0):
        super().__init__()
        # self.detector_margin = 0.25  # Border in percent of width(height): (x-m, y-m, x+m, y+m)

        self.input_res = input_res
        self.detector_margin = detector_margin
        self.detection_graph = tf.Graph()
        self.detection_sess = tf.Session(graph=self.detection_graph)

        # DET_PB_PATH = './farfaces2_python/models/fpn_v2_0.5_128_quantized/frozen_inference_graph.pb'
        with self.detection_graph.as_default():
            self.detection_graph, self.detection_sess = init_pb(pb_path)

            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            self.detection_boxes = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
            self.detection_scores = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
            self.detection_classes = tf.get_default_graph().get_tensor_by_name('detection_classes:0')
            self.num_detections = tf.get_default_graph().get_tensor_by_name('num_detections:0')

    def preprocessing(self, imgs):
        im_array = []
        for img in imgs:
            im_resized = cv2.resize(img, self.input_res)
            im_array.append(im_resized)
        return np.stack(im_array, axis=0)

    def __call__(self, imgs, detector_threshold=0.5, min_size_object=60):
        if not imgs:
            return []
        if not isinstance(imgs, list):
            im_pil = list(imgs)
        im_array = self.preprocessing(imgs)
        (boxes, scores, classes) = self.detection_sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes],
                feed_dict={self.image_tensor: im_array})
        res = []
        for idx, im in enumerate(imgs):
            im_width, im_height = im.shape[:2]
            filter_scores = scores[idx] >= detector_threshold
            batch_boxes = boxes[idx][filter_scores]
            margin_w = self.detector_margin / im_width
            margin_h = self.detector_margin / im_height
            batch_bboxes_coords = []
            for bbox in batch_boxes:
                x_min = max(bbox[1] - margin_w, 0)
                y_min = max(bbox[0] - margin_h, 0)
                x_max = min(bbox[3] + margin_w, 1.0)
                y_max = min(bbox[2] + margin_h, 1.0)
                bbox_coords = [x_min * im_width,
                               y_min * im_height,
                               x_max * im_width,
                               y_max * im_height]
                w_box = bbox_coords[2] - bbox_coords[0]
                h_box = bbox_coords[3] - bbox_coords[1]
                if (w_box >= min_size_object) and (h_box >= min_size_object):
                    batch_bboxes_coords.append(bbox_coords)
            res.append(np.array(batch_bboxes_coords, dtype=np.int))
        for res_bboxes in res:
            if res_bboxes.size != 0:
                return res
        return []