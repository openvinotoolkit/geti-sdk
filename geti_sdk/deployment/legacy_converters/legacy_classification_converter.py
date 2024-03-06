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
from typing import Dict, List, Optional, Tuple

from otx.algorithms.classification.utils import get_hierarchical_label_list
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
from otx.api.utils.labels_utils import get_empty_label


class ClassificationToAnnotationConverter(IPredictionToAnnotationConverter):
    """
    Convert Classification Predictions ModelAPI to Annotations.


    :param labels: Label Schema containing the label info of the task
    :param hierarchical_cls_heads_info: Info from model.hierarchical_info["cls_heads_info"]
    """

    def __init__(
        self,
        label_schema: LabelSchemaEntity,
        hierarchical_cls_heads_info: Optional[Dict] = None,
    ):
        if len(label_schema.get_labels(False)) == 1:
            self.labels = label_schema.get_labels(include_empty=True)
        else:
            self.labels = label_schema.get_labels(include_empty=False)
        self.empty_label = get_empty_label(label_schema)
        multilabel = len(label_schema.get_groups(False)) > 1
        multilabel = multilabel and len(label_schema.get_groups(False)) == len(
            label_schema.get_labels(include_empty=False)
        )
        self.hierarchical = not multilabel and len(label_schema.get_groups(False)) > 1

        self.label_schema = label_schema

        if self.hierarchical:
            self.labels = get_hierarchical_label_list(
                hierarchical_cls_heads_info, self.labels
            )

    def convert_to_annotation(
        self, predictions: List[Tuple[int, float]], metadata: Optional[Dict] = None
    ) -> AnnotationSceneEntity:
        """
        Convert predictions to OTX Annotation Scene using the metadata.

        :param predictions: Raw predictions from the model.
        :param metadata: Variable containing metadata information.
        :return: OTX annotation scene entity object.
        """
        labels = []
        for index, score in predictions:
            labels.append(ScoredLabel(self.labels[index], float(score)))
        if self.hierarchical:
            labels = self.label_schema.resolve_labels_probabilistic(labels)

        if not labels and self.empty_label:
            labels = [ScoredLabel(self.empty_label, probability=1.0)]

        annotations = [Annotation(Rectangle.generate_full_box(), labels=labels)]
        return AnnotationSceneEntity(
            kind=AnnotationSceneKind.PREDICTION, annotations=annotations
        )
