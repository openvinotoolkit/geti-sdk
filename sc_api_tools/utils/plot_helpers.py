from typing import Union, List

import numpy as np

import cv2

from sc_api_tools.data_models import Image, VideoFrame, Prediction, MediaList


def show_image_with_prediction(
        image: Union[Image, VideoFrame], prediction: Prediction
):
    """
    Displays an image with a prediction overlayed on top of it.

    :param image: Image to show prediction for
    :param prediction: Prediction to overlay on the image
    """
    alpha = 0.5
    mask = prediction.as_mask(image.media_information)
    result = np.uint8(
        image.numpy * alpha + mask[..., ::-1].copy() * (1 - alpha)
    )
    cv2.imshow(f'Prediction for {image.name}', result)
    cv2.waitKey(0)


def show_video_frames_with_predictions(
    video_frames: MediaList[VideoFrame],
    predictions: List[Prediction],
    wait_time: float = 3
):
    """
    Displays a list of VideoFrames, with their predictions overlayed on top. The
    parameter `wait_time` specifies the time each frame is shown, in seconds.

    :param video_frames: List of VideoFrames to show
    :param predictions: List of Predictions to overlay on the frames
    :param wait_time: Time to show each frame, in seconds
    """
    alpha = 0.5
    image_name = video_frames[0].name.split("_frame_")[0]
    for frame, prediction in zip(video_frames, predictions):
        mask = prediction.as_mask(frame.media_information)
        result = np.uint8(
            frame.numpy * alpha + mask[..., ::-1].copy() * (1 - alpha)
        )
        cv2.imshow(f'Prediction for {image_name}', result)
        cv2.waitKey(int(wait_time*1000))
