import cv2
import onnxruntime as ort
import time
import numpy as np
from .utils import letterbox
import os


class LivenessSpoof:
    def __init__(self,
                 weights: str = None,
                 input_res: tuple = (224, 224),
                 batch_size: int = 1,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.weights = weights
        self.input_res = input_res
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.ort_session, self.input_name = self._init_session_(self.weights)
        self.spoof_type = {0:'Live', 1:'Photo', 2:'Poster', 3:'A4', 4:'Face Mask', 5:'Upper Body Mask',
                      6:'Region Mask', 7:'PC', 8:'Pad', 9:'Phone', 10: '3D Mask'}

    def _init_session_(self, path_onnx_model: str):
        ort_session = None
        input_name = None
        if os.path.isfile(path_onnx_model):
            ort_session = ort.InferenceSession(path_onnx_model)
            input_name = ort_session.get_inputs()[0].name
        return ort_session, input_name

    def preprocessing(self, img, img_size=(224, 224), mean=(0.5, 0.5, 0.5), std=(0.225, 0.225, 0.225)):
        img_input, ratio, (dw, dh) = letterbox(img, new_shape=img_size, color=(0, 0, 0))
        img_input = img_input / 255.0
        for i_chanel in range(len(mean)):
            img_input[:, :, i_chanel] = (img_input[:, :, i_chanel] - mean[i_chanel]) / std[i_chanel]
        img_input = img_input.transpose(2, 0, 1)
        img_input = np.ascontiguousarray(img_input)
        img_input = img_input.astype(np.float32)
        img_input = np.expand_dims(img_input, axis=0)
        return img_input, ratio, (dw, dh)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def postprocessing(self, result_onnx):
        class_id = self.softmax(result_onnx).argmax()
        return class_id

    def __call__(self, img, ):
        if not self.ort_session:
            return False
        img_input, ratio, (dw, dh) = self.preprocessing(img,
                                                        img_size=self.input_res,
                                                        mean=self.mean,
                                                        std=self.std)
        onnx_result = self.ort_session.run([], {self.input_name: img_input})
        res = onnx_result[0].squeeze()
        class_id = self.postprocessing(res)
        return class_id

