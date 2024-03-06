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

"""Module implements the Postprocessor class."""

from typing import Dict, List, Tuple

import numpy as np
import otx
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    IPredictionToAnnotationConverter,
    create_converter,
)

from geti_sdk.data_models.annotations import Annotation
from geti_sdk.data_models.enums.task_type import TaskType
from geti_sdk.data_models.label import ScoredLabel
from geti_sdk.data_models.predictions import Prediction
from geti_sdk.data_models.shapes import Polygon, Rectangle, RotatedRectangle
from geti_sdk.data_models.task import Task
from geti_sdk.deployment.legacy_converters.legacy_anomaly_converter import (
    AnomalyClassificationToAnnotationConverter,
)


def detection2array(detections: List) -> np.ndarray:
    """
    Convert list of OpenVINO Detection to a numpy array.

    :param detections: List of OpenVINO Detection containing score, id, xmin, ymin, xmax, ymax

    :return: np.ndarray: numpy array with [label, confidence, x1, y1, x2, y2]
    """
    scores = np.empty((0, 1), dtype=np.float32)
    labels = np.empty((0, 1), dtype=np.uint32)
    boxes = np.empty((0, 4), dtype=np.float32)
    for det in detections:
        if (det.xmax - det.xmin) * (det.ymax - det.ymin) < 1.0:
            continue
        scores = np.append(scores, [[det.score]], axis=0)
        labels = np.append(labels, [[det.id]], axis=0)
        boxes = np.append(
            boxes,
            [[float(det.xmin), float(det.ymin), float(det.xmax), float(det.ymax)]],
            axis=0,
        )
    detections = np.concatenate((labels, scores, boxes), -1)
    return detections


class Postprocessor:
    """
    Postprocessor class responsible for converting the output of the model to a Prediction object.

    :param labels: Label schema to be used for the conversion.
    :param configuration: Configuration to be used for the conversion.
    :param task: Task object containing the task metadata.
    """

    def __init__(self, label_schema, configuration, task: Task) -> None:
        self.task = task
        self.ote_label_schema = label_schema

        # Create OTX converter
        converter_args = {"labels": self.ote_label_schema}
        if otx.__version__ > "1.2.0":
            if "use_ellipse_shapes" not in configuration.keys():
                configuration.update({"use_ellipse_shapes": False})
            converter_args["configuration"] = configuration

        self.converter: IPredictionToAnnotationConverter = create_converter(
            converter_type=self.task.type.to_ote_domain(), **converter_args
        )

    def __call__(
        self,
        postprocessing_results: List,
        image: np.ndarray,
        metadata: Dict[str, Tuple[int, int, int]],
    ) -> Prediction:
        """
        Convert the postprocessing results to a Prediction object.
        """
        # Handle empty annotations
        if isinstance(postprocessing_results, (np.ndarray, list)):
            try:
                n_outputs = len(postprocessing_results)
            except TypeError:
                n_outputs = 1
        else:
            # Handle the new modelAPI output formats for detection and instance
            # segmentation models
            if (
                hasattr(postprocessing_results, "objects")
                and self.task.type == TaskType.DETECTION
            ):
                n_outputs = len(postprocessing_results.objects)
                postprocessing_results = detection2array(postprocessing_results.objects)
            elif hasattr(
                postprocessing_results, "segmentedObjects"
            ) and self.task.type in [
                TaskType.INSTANCE_SEGMENTATION,
                TaskType.ROTATED_DETECTION,
            ]:
                n_outputs = len(postprocessing_results.segmentedObjects)
                postprocessing_results = postprocessing_results.segmentedObjects
            elif isinstance(postprocessing_results, tuple):
                try:
                    n_outputs = len(postprocessing_results)
                except TypeError:
                    n_outputs = 1
            else:
                raise ValueError(
                    f"Unknown postprocessing output of type "
                    f"`{type(postprocessing_results)}` for task `{self.task.title}`."
                )

        # Proceed with postprocessing
        width: int = image.shape[1]
        height: int = image.shape[0]

        if n_outputs != 0:
            try:
                annotation_scene_entity = self.converter.convert_to_annotation(
                    predictions=postprocessing_results, metadata=metadata
                )
            except AttributeError:
                # Add backwards compatibility for anomaly models created in Geti v1.8 and below
                if self.task.type.is_anomaly:
                    legacy_converter = AnomalyClassificationToAnnotationConverter(
                        label_schema=self.ote_label_schema
                    )
                    annotation_scene_entity = legacy_converter.convert_to_annotation(
                        predictions=postprocessing_results, metadata=metadata
                    )
                    self.converter = legacy_converter

            prediction = Prediction.from_ote(
                annotation_scene_entity, image_width=width, image_height=height
            )
        else:
            prediction = Prediction(annotations=[])

        # print(
        #     "pre-converter",
        #     postprocessing_results,
        #     "metadata",
        #     metadata,
        # )
        # print("width", width, "height", height)
        # print("post-converter", prediction)

        # Empty label is not generated by OTE correctly, append it here if there are
        # no other predictions
        if len(prediction.annotations) == 0:
            empty_label = next(
                (label for label in self.task.labels if label.is_empty), None
            )
            if empty_label is not None:
                prediction.append(
                    Annotation(
                        shape=Rectangle(x=0, y=0, width=width, height=height),
                        labels=[ScoredLabel.from_label(empty_label, probability=1)],
                    )
                )

        # Rotated detection models produce Polygons, convert them here to
        # RotatedRectangles
        if self.task.type == TaskType.ROTATED_DETECTION:
            for annotation in prediction.annotations:
                if isinstance(annotation.shape, Polygon):
                    annotation.shape = RotatedRectangle.from_polygon(annotation.shape)

        return prediction
