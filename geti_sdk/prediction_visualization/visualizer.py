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
from geti_sdk.data_models.predictions import Prediction
from geti_sdk.prediction_visualization.shape_drawer import ShapeDrawer


class Visualizer:
    """
    Visualize the predicted output by drawing the annotations on the input image.

    :example:
        >>> prediction = deployment.infer(image)
        >>> visualizer = Visualizer()
        >>> output_image = visualizer.draw(image, prediction )
        >>> Display(output_image)

    """

    def __init__(
        self,
        window_name: Optional[str] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_count: bool = False,
        is_one_label: bool = False,
        delay: Optional[int] = None,
        output: Optional[str] = None,
    ) -> None:
        """
        Initialize the Visualizer.

        :param window_name: Name of the window to be shown
        :param show_labels: Show labels on the output image
        :param show_confidence: Show confidence on the output image
        :param show_count: Show count of the shapes on the output image
        :param is_one_label: Show only one label on the output image
        :param delay: Delay time for the output image
        :param output: Path to save the output image
        """
        self.window_name = "Window" if window_name is None else window_name
        self.shape_drawer = ShapeDrawer(
            show_count, is_one_label, show_labels, show_confidence
        )

        self.delay = delay
        if delay is None:
            self.delay = 1
        self.output = output

    def draw(
        self,
        image: np.ndarray,
        annotation: AnnotationScene,
        fill_shapes: bool = True,
        confidence_threshold: Optional[float] = None,
        meta: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Draw annotations on the image.

        :param image: Input image in RGB format
        :param annotation: Annotations to be drawn on the input image
        :param meta: Optional meta information
        :param fill_shapes: Fill shapes with color
        :param confidence_threshold: Confidence threshold to filter annotations.
            Must be in range [0, 1].
        :return: Output image with annotations in RGB format
        """
        if confidence_threshold is not None:
            annotation = annotation.filter_by_confidence(confidence_threshold)
        result = self.shape_drawer.draw(
            image, annotation, labels=[], fill_shapes=fill_shapes
        )
        return result

    def explain_label(
        self,
        image: np.ndarray,
        prediction: Prediction,
        label_name: str,
        opacity: float = 0.5,
        show_predictions: bool = True,
    ) -> np.ndarray:
        """
        Draw saliency map overlay on the image.

        :param image: Input image in RGB format
        :param prediction: Prediction object containing saliency maps
        :param label_name: Label name to be explained
        :param opacity: Opacity of the saliency map overlay
        :param show_predictions: Show predictions for the label on the output image
        :return: Output image with saliency map overlay in RGB format
        """
        saliency_map = None
        for pred_map in prediction.maps:
            if pred_map.type == "saliency map":
                saliency_map = pred_map.data
                break
        if saliency_map is None:
            raise ValueError("Prediction does not contain saliency maps")
        if label_name not in saliency_map:
            raise ValueError(
                f"Saliency map for label {label_name} is not found in the prediction."
            )
        # Accessing the saliency map for the label
        saliency_map = saliency_map[label_name]
        if saliency_map.shape[:2] != image.shape[:2]:
            saliency_map = cv2.resize(
                saliency_map,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
        # Visualizing the saliency map
        overlay = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        result = cv2.addWeighted(image, 1 - opacity, overlay, opacity, 0)
        if show_predictions:
            filtered_prediction = prediction.filter_annotations([label_name])
            result = self.draw(result, filtered_prediction, fill_shapes=False)
        return result

    def show(self, image: np.ndarray) -> None:
        """
        Show result image.

        :param image: Image to be shown (in RGB order).
        """
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, image_bgr)

    def is_quit(self) -> bool:
        """Check user wish to quit."""
        return ord("q") == cv2.waitKey(self.delay)
