from typing import Union, List

import numpy as np

import cv2

from sc_api_tools.data_models.annotations import AnnotationScene
from sc_api_tools.data_models.media import Image, VideoFrame
from sc_api_tools.data_models.predictions import Prediction
from sc_api_tools.data_models.containers import MediaList


def show_image_with_annotation_scene(
        image: Union[Image, VideoFrame],
        annotation_scene: Union[AnnotationScene, Prediction]
):
    """
    Displays an image with an annotation_scene overlayed on top of it.

    :param image: Image to show prediction for
    :param annotation_scene: Annotations or Predictions to overlay on the image
    """
    alpha = 0.5
    mask = annotation_scene.as_mask(image.media_information)
    result = np.uint8(
        image.numpy * alpha + mask[..., ::-1].copy() * (1 - alpha)
    )
    cv2.imshow(f'Prediction for {image.name}', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_video_frames_with_annotation_scenes(
    video_frames: MediaList[VideoFrame],
    annotation_scenes: List[Union[AnnotationScene, Prediction]],
    wait_time: float = 3
):
    """
    Displays a list of VideoFrames, with their annotations or predictions overlayed on
    top. The parameter `wait_time` specifies the time each frame is shown, in seconds.

    :param video_frames: List of VideoFrames to show
    :param annotation_scenes: List of AnnotationsScenes or Predictions to overlay on
        the frames
    :param wait_time: Time to show each frame, in seconds
    """
    alpha = 0.5
    image_name = video_frames[0].name.split("_frame_")[0]
    for frame, annotation_scene in zip(video_frames, annotation_scenes):
        mask = annotation_scene.as_mask(frame.media_information)
        result = np.uint8(
            frame.numpy * alpha + mask[..., ::-1].copy() * (1 - alpha)
        )
        cv2.imshow(f'Prediction for {image_name}', result)
        cv2.waitKey(int(wait_time*1000))
        cv2.destroyAllWindows()
