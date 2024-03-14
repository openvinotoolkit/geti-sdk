# Copyright (C) 2024 Intel Corporation
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

"""The package provides the Visualizer class for models predictions visualization."""


from typing import Optional

import cv2
import numpy as np

from geti_sdk.data_models.annotation_scene import AnnotationScene
from geti_sdk.prediction_visualization.shape_drawer import ShapeDrawer


class Visualizer:
    """
    Visualize the predicted output by drawing the annotations on the input image.

    :example:
        >>> predictions = inference_model.predict(frame)
        >>> annotation = prediction_converter.convert_to_annotation(predictions)
        >>> output = visualizer.draw(frame, annotation.shape, annotation.get_labels())
        >>> visualizer.show(output)

    """

    def __init__(
        self,
        window_name: Optional[str] = None,
        show_count: bool = False,
        is_one_label: bool = False,
        no_show: bool = False,
        delay: Optional[int] = None,
        output: Optional[str] = None,
    ) -> None:
        self.window_name = "Window" if window_name is None else window_name
        self.shape_drawer = ShapeDrawer(show_count, is_one_label)

        self.delay = delay
        self.no_show = no_show
        if delay is None:
            self.delay = 1
        self.output = output

    def draw(
        self,
        image: np.ndarray,
        annotation: AnnotationScene,
        meta: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Draw annotations on the image.

        :param image: Input image
        :param annotation: Annotations to be drawn on the input image
        :return: Output image with annotations.
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return self.shape_drawer.draw(image, annotation, labels=[])

    def show(self, image: np.ndarray) -> None:
        """
        Show result image.

        :param image: Image to be shown.
        """
        if not self.no_show:
            cv2.imshow(self.window_name, image)

    def is_quit(self) -> bool:
        """Check user wish to quit."""
        if self.no_show:
            return False

        return ord("q") == cv2.waitKey(self.delay)
