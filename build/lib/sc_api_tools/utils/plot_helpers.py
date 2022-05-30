from typing import Union, List

import numpy as np

import cv2

from sc_api_tools.data_models.annotation_scene import AnnotationScene
from sc_api_tools.data_models.media import Image, VideoFrame
from sc_api_tools.data_models.media import MediaInformation
from sc_api_tools.data_models.predictions import Prediction
from sc_api_tools.data_models.containers import MediaList


def show_image_with_annotation_scene(
        image: Union[Image, VideoFrame, np.ndarray],
        annotation_scene: Union[AnnotationScene, Prediction]
):
    """
    Displays an image with an annotation_scene overlayed on top of it.

    :param image: Image to show prediction for
    :param annotation_scene: Annotations or Predictions to overlay on the image
    """
    if type(annotation_scene) == AnnotationScene:
        plot_type = 'Annotation'
    elif type(annotation_scene) == Prediction:
        plot_type = 'Prediction'
    else:
        raise ValueError(
            f"Invalid input: Unable to plot object of type {type(annotation_scene)}."
        )
    if isinstance(image, np.ndarray):
        media_information = MediaInformation(
            "", height=image.shape[0], width=image.shape[1]
        )
        name = 'Numpy image'
    else:
        media_information = image.media_information
        name = image.name
    mask = annotation_scene.as_mask(media_information)

    if isinstance(image, np.ndarray):
        result = image.copy()
    else:
        result = image.numpy.copy()
    result[np.sum(mask, axis=-1) > 0] = 0
    result += mask[..., ::-1]

    cv2.imshow(f'{plot_type} for {name}', result)
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
    image_name = video_frames[0].name.split("_frame_")[0]
    for frame, annotation_scene in zip(video_frames, annotation_scenes):
        if type(annotation_scene) == AnnotationScene:
            name = 'Annotation'
        elif type(annotation_scene) == Prediction:
            name = 'Prediction'
        else:
            raise ValueError(
                f"Invalid input: Unable to plot object of type "
                f"{type(annotation_scene)}."
            )
        mask = annotation_scene.as_mask(frame.media_information)

        result = frame.numpy.copy()
        result[np.sum(mask, axis=-1) > 0] = 0
        result += mask[..., ::-1]

        cv2.imshow(f'{name} for {image_name}', result)
        cv2.waitKey(int(wait_time*1000))
        cv2.destroyAllWindows()
