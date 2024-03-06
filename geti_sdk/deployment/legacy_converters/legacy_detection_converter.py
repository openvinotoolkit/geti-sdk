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
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    IPredictionToAnnotationConverter,
)


class RotatedRectToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Convert Rotated Rect (mask) Predictions ModelAPI to Annotations.

    :param labels: Label Schema containing the label info of the task
    """

    def __init__(
        self, labels: LabelSchemaEntity, configuration: Optional[Dict[str, Any]] = None
    ):
        self.labels = labels.get_labels(include_empty=False)
        self.use_ellipse_shapes = False
        self.confidence_threshold = 0.0
        if configuration is not None:
            if "use_ellipse_shapes" in configuration:
                self.use_ellipse_shapes = configuration["use_ellipse_shapes"]
            if "confidence_threshold" in configuration:
                self.confidence_threshold = configuration["confidence_threshold"]

    def convert_to_annotation(
        self, predictions: tuple, metadata: Dict[str, Any]
    ) -> AnnotationSceneEntity:
        """
        Convert predictions to OTX Annotation Scene using the metadata.

        :param predictions: Raw predictions from the model.
        :param metadata: Variable containing metadata information.
        :return: OTX annotation scene entity object.
        """
        annotations = []
        height, width, _ = metadata["original_shape"]
        shape: Union[Polygon, Ellipse]
        for score, class_idx, box, mask in zip(*predictions):
            if score < self.confidence_threshold:
                continue
            if self.use_ellipse_shapes:
                shape = Ellipse(
                    box[0] / width, box[1] / height, box[2] / width, box[3] / height
                )
                annotations.append(
                    Annotation(
                        shape,
                        labels=[
                            ScoredLabel(self.labels[int(class_idx) - 1], float(score))
                        ],
                    )
                )
            else:
                mask = mask.astype(np.uint8)
                contours, hierarchies = cv2.findContours(
                    mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                )
                if hierarchies is None:
                    continue
                for contour, hierarchy in zip(contours, hierarchies[0]):
                    if hierarchy[3] != -1:
                        continue
                    if len(contour) <= 2 or cv2.contourArea(contour) < 1.0:
                        continue
                    points = [
                        Point(
                            x=point[0] / width,
                            y=point[1] / height,
                        )
                        for point in cv2.boxPoints(cv2.minAreaRect(contour))
                    ]
                    shape = Polygon(points=points)
                    annotations.append(
                        Annotation(
                            shape,
                            labels=[
                                ScoredLabel(
                                    self.labels[int(class_idx) - 1], float(score)
                                )
                            ],
                        )
                    )
        annotation_scene = AnnotationSceneEntity(
            kind=AnnotationSceneKind.PREDICTION,
            annotations=annotations,
        )
        return annotation_scene
