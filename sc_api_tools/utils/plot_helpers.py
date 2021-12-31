from typing import Union

import numpy as np

import cv2
from sc_api_tools.data_models import Image, VideoFrame, Prediction


def show_image_with_prediction(image: Union[Image, VideoFrame], prediction: Prediction):
    alpha = 0.5
    mask = prediction.as_mask(image.media_information)
    result = np.uint8(
        image.numpy * alpha + mask[..., ::-1].copy() * (1 - alpha)
    )
    cv2.imshow(f'Prediction for {image.name}', result)
    cv2.waitKey(0)
