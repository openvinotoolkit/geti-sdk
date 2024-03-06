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
from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    IPredictionToAnnotationConverter,
)
from otx.api.utils.segmentation_utils import create_annotation_from_segmentation_map


class SegmentationToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Convert Segmentation Predictions ModelAPI to Annotations.

    :param labels: Label Schema containing the label info of the task
    """

    def __init__(self, label_schema: LabelSchemaEntity):
        labels = label_schema.get_labels(include_empty=False)
        self.label_map = dict(enumerate(labels, 1))

    def convert_to_annotation(
        self, predictions: np.ndarray, metadata: Dict[str, Any]
    ) -> AnnotationSceneEntity:
        """
        Convert predictions to OTX Annotation Scene using the metadata.

        :param predictions: Raw predictions from the model.
        :param metadata: Variable containing metadata information.
        :return: OTX annotation scene entity object.
        """
        soft_prediction = metadata.get("soft_prediction", np.ones(predictions.shape))
        annotations = create_annotation_from_segmentation_map(
            hard_prediction=predictions,
            soft_prediction=soft_prediction,
            label_map=self.label_map,
        )

        return AnnotationSceneEntity(
            kind=AnnotationSceneKind.PREDICTION, annotations=annotations
        )
