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
from typing import Any, Dict

import numpy as np
from openvino.model_api.models.utils import AnomalyResult
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    IPredictionToAnnotationConverter,
)
from otx.api.utils.segmentation_utils import create_annotation_from_segmentation_map


class AnomalyClassificationToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Convert AnomalyClassification Predictions ModelAPI to Annotations.

    :param labels: Label Schema containing the label info of the task
    """

    def __init__(self, label_schema: LabelSchemaEntity):
        labels = label_schema.get_labels(include_empty=False)
        self.normal_label = [label for label in labels if not label.is_anomalous][0]
        self.anomalous_label = [label for label in labels if label.is_anomalous][0]

    def convert_to_annotation(
        self, predictions: np.ndarray, metadata: Dict[str, Any]
    ) -> AnnotationSceneEntity:
        """
        Convert predictions to OTX Annotation Scene using the metadata.

        :param predictions: Raw predictions from the model.
        :param metadata: Variable containing metadata information.
        :return: OTX annotation scene entity object.
        """
        pred_label = predictions >= metadata.get("threshold", 0.5)

        label = self.anomalous_label if pred_label else self.normal_label
        probability = (1 - predictions) if predictions < 0.5 else predictions

        annotations = [
            Annotation(
                Rectangle.generate_full_box(),
                labels=[ScoredLabel(label=label, probability=float(probability))],
            )
        ]
        return AnnotationSceneEntity(
            kind=AnnotationSceneKind.PREDICTION, annotations=annotations
        )


class AnomalySegmentationToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Convert AnomalySegmentation Predictions ModelAPI to Annotations.

    :param labels: Label Schema containing the label info of the task
    """

    def __init__(self, label_schema: LabelSchemaEntity):
        labels = label_schema.get_labels(include_empty=False)
        self.normal_label = [label for label in labels if not label.is_anomalous][0]
        self.anomalous_label = [label for label in labels if label.is_anomalous][0]
        self.label_map = {0: self.normal_label, 1: self.anomalous_label}

    def convert_to_annotation(
        self, predictions: AnomalyResult, metadata: Dict[str, Any]
    ) -> AnnotationSceneEntity:
        """
        Convert predictions to OTX Annotation Scene using the metadata.

        :param predictions: Raw predictions from the model.
        :param metadata: Variable containing metadata information.
        :return: OTX annotation scene entity object.
        """
        assert predictions.pred_mask is not None
        assert predictions.anomaly_map is not None
        annotations = create_annotation_from_segmentation_map(
            predictions.pred_mask, predictions.anomaly_map / 255.0, self.label_map
        )
        if len(annotations) == 0:
            # TODO: add confidence to this label
            annotations = [
                Annotation(
                    Rectangle.generate_full_box(),
                    labels=[ScoredLabel(label=self.normal_label, probability=1.0)],
                )
            ]
        return AnnotationSceneEntity(
            kind=AnnotationSceneKind.PREDICTION, annotations=annotations
        )


class AnomalyDetectionToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Convert Anomaly Detection Predictions ModelAPI to Annotations.

    :param labels: Label Schema containing the label info of the task
    """

    def __init__(self, label_schema: LabelSchemaEntity):
        labels = label_schema.get_labels(include_empty=False)
        self.normal_label = [label for label in labels if not label.is_anomalous][0]
        self.anomalous_label = [label for label in labels if label.is_anomalous][0]
        self.label_map = {0: self.normal_label, 1: self.anomalous_label}

    def convert_to_annotation(
        self, predictions: AnomalyResult, metadata: Dict[str, Any]
    ) -> AnnotationSceneEntity:
        """
        Convert predictions to OTX Annotation Scene using the metadata.

        :param predictions: Raw predictions from the model.
        :param metadata: Variable containing metadata information.
        :return: OTX annotation scene entity object.
        """
        assert predictions.pred_boxes is not None
        assert predictions.pred_score is not None
        assert predictions.pred_mask is not None
        annotations = []
        image_h, image_w = predictions.pred_mask.shape
        for box in predictions.pred_boxes:
            annotations.append(
                Annotation(
                    Rectangle(
                        box[0] / image_w,
                        box[1] / image_h,
                        box[2] / image_w,
                        box[3] / image_h,
                    ),
                    labels=[
                        ScoredLabel(
                            label=self.anomalous_label,
                            probability=predictions.pred_score,
                        )
                    ],
                )
            )
        if len(annotations) == 0:
            # TODO: add confidence to this label
            annotations = [
                Annotation(
                    Rectangle.generate_full_box(),
                    labels=[ScoredLabel(label=self.normal_label, probability=1.0)],
                )
            ]
        return AnnotationSceneEntity(
            kind=AnnotationSceneKind.PREDICTION, annotations=annotations
        )
