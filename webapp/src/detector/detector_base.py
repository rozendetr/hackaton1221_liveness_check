import numpy as np


class Detector:
    def __init__(self):
        pass

    def __call__(self, image, detector_threshold, min_size_object) -> np.ndarray:
        raise NotImplementedError