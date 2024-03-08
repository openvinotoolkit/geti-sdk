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

from typing import List, Tuple

import numpy as np
import otx
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.model_template import Domain as OteDomain
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    IPredictionToAnnotationConverter,
    create_converter,
)

from geti_sdk.data_models.annotations import Annotation
from geti_sdk.data_models.enums.domain import Domain
from geti_sdk.data_models.enums.task_type import TaskType
from geti_sdk.data_models.label import ScoredLabel
from geti_sdk.data_models.label_schema import LabelSchema
from geti_sdk.data_models.predictions import Prediction
from geti_sdk.data_models.shapes import Polygon, Rectangle, RotatedRectangle
from geti_sdk.deployment.legacy_converters.legacy_anomaly_converter import (
    AnomalyClassificationToAnnotationConverter,
)
from geti_sdk.deployment.predictions_postprocessing.utils.detection_utils import (
    detection2array,
)


class LegacyConverter:
    """
    LegacyConverter class responsible for converting the output of the model to a Prediction object.
    For models generated with Geti v1.8 and below.

    :param labels: Label schema to be used for the conversion.
    :param configuration: Configuration to be used for the conversion.
    :param task: Task object containing the task metadata.
    """

    def __init__(
        self, label_schema: LabelSchema, configuration, domain: Domain
    ) -> None:
        self.domain = domain
        self.task_type = TaskType[self.domain.name]
        self.label_schema = LabelSchemaEntity(
            label_groups=[
                LabelGroup(
                    name=group.name,
                    labels=[label.to_ote(self.task_type) for label in group.labels],
                    group_type=LabelGroupType[group.group_type.name],
                    id=group.id,
                )
                for group in label_schema.get_groups(include_empty=True)
            ]
        )

        # Create OTX converter
        converter_args = {"labels": self.label_schema}
        if otx.__version__ > "1.2.0":
            if "use_ellipse_shapes" not in configuration.keys():
                configuration.update({"use_ellipse_shapes": False})
            converter_args["configuration"] = configuration

        self.converter: IPredictionToAnnotationConverter = create_converter(
            converter_type=OteDomain[self.domain.name], **converter_args
        )

    def convert_to_prediction(
        self, postprocessing_results: List, image_shape: Tuple[int], **kwargs
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
                and self.domain == Domain.DETECTION
            ):
                n_outputs = len(postprocessing_results.objects)
                postprocessing_results = detection2array(postprocessing_results.objects)
            elif hasattr(
                postprocessing_results, "segmentedObjects"
            ) and self.domain in [
                Domain.INSTANCE_SEGMENTATION,
                Domain.ROTATED_DETECTION,
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
        width: int = image_shape[1]
        height: int = image_shape[0]

        if n_outputs != 0:
            try:
                annotation_scene_entity = self.converter.convert_to_annotation(
                    predictions=postprocessing_results,
                    metadata={"original_shape": image_shape},
                )
            except AttributeError:
                # Add backwards compatibility for anomaly models created in Geti v1.8 and below
                if self.domain == Domain.ANOMALY_CLASSIFICATION:
                    legacy_converter = AnomalyClassificationToAnnotationConverter(
                        label_schema=self.label_schema
                    )
                    annotation_scene_entity = legacy_converter.convert_to_annotation(
                        predictions=postprocessing_results,
                        metadata={"original_shape": image_shape},
                    )
                    self.converter = legacy_converter

            prediction = Prediction.from_ote(
                annotation_scene_entity, image_width=width, image_height=height
            )
        else:
            prediction = Prediction(annotations=[])

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
        if self.domain == Domain.ROTATED_DETECTION:
            for annotation in prediction.annotations:
                if isinstance(annotation.shape, Polygon):
                    annotation.shape = RotatedRectangle.from_polygon(annotation.shape)

        return prediction
