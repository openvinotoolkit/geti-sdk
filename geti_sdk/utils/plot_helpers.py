# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import List, Optional, Union

import cv2
import numpy as np
from IPython.display import display
from ote_sdk.usecases.exportable_code.visualizers import Visualizer
from PIL import Image as PILImage

from geti_sdk.data_models.annotation_scene import AnnotationScene
from geti_sdk.data_models.containers import MediaList
from geti_sdk.data_models.media import Image, MediaInformation, VideoFrame
from geti_sdk.data_models.predictions import Prediction


def show_image_with_annotation_scene(
    image: Union[Image, VideoFrame, np.ndarray],
    annotation_scene: Union[AnnotationScene, Prediction],
    filepath: Optional[str] = None,
    show_in_notebook: bool = False,
    show_results: bool = True,
) -> np.ndarray:
    """
    Display an image with an annotation_scene overlayed on top of it.

    :param image: Image to show prediction for.
        NOTE: `image` is expected to have R,G,B channel ordering
    :param annotation_scene: Annotations or Predictions to overlay on the image
    :param filepath: Optional filepath to save the image with annotation overlay to.
        If left as None, the result will not be saved to file
    :param show_in_notebook: True if the image needs to be shown in a notebook context.
        Setting this to True will display the image inline in the notebook. Setting it
        to False will open a pop up to show the image.
    :param show_results: True to show the results. If `show_in_notebook` is True, this
        will display the image with the annotations inside the notebook. If
        `show_in_notebook` is False, a new opencv window will pop up. If
        `show_results` is set to False, the results will not be shown but will only
        be returned instead
    """
    if type(annotation_scene) == AnnotationScene:
        plot_type = "Annotation"
    elif type(annotation_scene) == Prediction:
        plot_type = "Prediction"
    else:
        raise ValueError(
            f"Invalid input: Unable to plot object of type {type(annotation_scene)}."
        )
    if isinstance(image, np.ndarray):
        media_information = MediaInformation(
            "", height=image.shape[0], width=image.shape[1]
        )
        name = "Numpy image"
    else:
        media_information = image.media_information
        name = image.name

    window_name = f"{plot_type} for {name}"
    visualizer = Visualizer(window_name=window_name)
    ote_annotation_scene = annotation_scene.to_ote(
        image_width=media_information.width, image_height=media_information.height
    )

    if isinstance(image, np.ndarray):
        numpy_image = image.copy()
    else:
        numpy_image = image.numpy.copy()

    result = visualizer.draw(image=numpy_image, annotation=ote_annotation_scene)

    if filepath is None:
        if show_results:
            if not show_in_notebook:
                cv2.imshow(window_name, result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            else:
                image = PILImage.fromarray(result)
                display(image)
    else:
        cv2.imwrite(filepath, result)

    return result


def show_video_frames_with_annotation_scenes(
    video_frames: MediaList[VideoFrame],
    annotation_scenes: List[Union[AnnotationScene, Prediction]],
    wait_time: float = 1,
    filepath: Optional[str] = None,
):
    """
    Display a list of VideoFrames, with their annotations or predictions overlayed on
    top. The parameter `wait_time` specifies the time each frame is shown, in seconds.

    :param video_frames: List of VideoFrames to show
    :param annotation_scenes: List of AnnotationsScenes or Predictions to overlay on
        the frames
    :param wait_time: Time to show each frame, in seconds
    :param filepath: Optional filepath to save the video with annotation overlay to.
        If left as None, the video frames will be shown in a new opencv window
    """
    first_frame = video_frames[0]

    out_writer: Optional[cv2.VideoWriter] = None
    if filepath is not None:
        out_writer = cv2.VideoWriter(
            filename=f"{filepath}",
            fourcc=cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            fps=1 / wait_time,
            frameSize=(
                first_frame.media_information.width,
                first_frame.media_information.height,
            ),
        )

    for frame, annotation_scene in zip(video_frames, annotation_scenes):
        if type(annotation_scene) == AnnotationScene:
            name = "Annotation"
        elif type(annotation_scene) == Prediction:
            name = "Prediction"
        else:
            raise ValueError(
                f"Invalid input: Unable to plot object of type "
                f"{type(annotation_scene)}."
            )
        ote_annotation = annotation_scene.to_ote(
            image_width=frame.media_information.width,
            image_height=frame.media_information.height,
        )

        numpy_frame = frame.numpy.copy()
        window_name = f"{name} for '{frame.video_name}'"
        visualizer = Visualizer(window_name=window_name)
        result = visualizer.draw(numpy_frame, annotation=ote_annotation)

        if out_writer is None:
            cv2.imshow(window_name, result)
            cv2.waitKey(int(wait_time * 1000))
        else:
            out_writer.write(result)

    if out_writer is None:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    else:
        out_writer.release()
