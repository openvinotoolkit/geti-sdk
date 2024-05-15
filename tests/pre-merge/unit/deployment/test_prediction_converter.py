# INTEL CONFIDENTIAL
#
# Copyright (C) 2024 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.
from typing import Tuple

import numpy as np
import pytest
from model_api.models.utils import (
    AnomalyResult,
    ClassificationResult,
    Detection,
    DetectionResult,
    ImageResultWithSoftPrediction,
    InstanceSegmentationResult,
    SegmentedObject,
)

from geti_sdk.data_models.enums.domain import Domain
from geti_sdk.data_models.label import ScoredLabel
from geti_sdk.data_models.label_group import LabelGroup
from geti_sdk.data_models.label_schema import LabelSchema
from geti_sdk.data_models.shapes import (
    Ellipse,
    Point,
    Polygon,
    Rectangle,
    RotatedRectangle,
)
from geti_sdk.deployment.predictions_postprocessing.results_converter.results_to_prediction_converter import (
    AnomalyToPredictionConverter,
    ClassificationToPredictionConverter,
    DetectionToPredictionConverter,
    RotatedRectToPredictionConverter,
    SegmentationToPredictionConverter,
)


def coords_to_xmin_xmax_width_height(
    coords: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    "Convert bbox to xmin, ymin, width, height format"
    x1, y1, x2, y2 = coords
    return x1, y1, x2 - x1, y2 - y1


@pytest.mark.JobsComponent
class TestInferenceResultsToPredictionConverter:
    def test_classification_to_prediction_converter(self, fxt_label_schema_factory):
        # Arrange
        label_schema = fxt_label_schema_factory(Domain.CLASSIFICATION)
        labels = label_schema.get_labels(include_empty=False)
        raw_prediction = ClassificationResult(
            top_labels=[(1, labels[1].name, 0.81)],
            raw_scores=[0.19, 0.81],
            saliency_map=None,
            feature_vector=None,
        )

        # Act
        converter = ClassificationToPredictionConverter(label_schema)
        prediction = converter.convert_to_prediction(
            raw_prediction, image_shape=(10, 10)
        )

        # Assert
        assert converter.labels == labels
        assert len(prediction.annotations) == 1
        predicted_label = prediction.annotations[0].labels[0]
        assert predicted_label.name == labels[1].name
        assert predicted_label.probability == 0.81

    @pytest.mark.parametrize("use_ellipse_shapes", [True, False])
    def test_detection_to_prediction_converter(
        self, use_ellipse_shapes, fxt_label_schema_factory
    ):
        # Arrange
        label_schema = fxt_label_schema_factory(Domain.DETECTION)
        labels = label_schema.get_labels(include_empty=False)
        coords = [12.0, 41.0, 12.5, 45.5]
        raw_prediction = DetectionResult(
            objects=[Detection(*coords, score=0.51, id=0)],
            saliency_map=None,
            feature_vector=None,
        )

        # Act
        converter = DetectionToPredictionConverter(
            label_schema=label_schema,
            configuration={"use_ellipse_shapes": use_ellipse_shapes},
        )
        prediction = converter.convert_to_prediction(raw_prediction)

        # Assert
        assert converter.labels == labels
        assert len(prediction.annotations) == 1
        if use_ellipse_shapes:
            assert prediction.annotations[0].shape == Ellipse(
                *coords_to_xmin_xmax_width_height(coords)
            )
        else:
            assert prediction.annotations[0].shape == Rectangle(
                *coords_to_xmin_xmax_width_height(coords)
            )
        assert prediction.annotations[0].labels[0] == ScoredLabel.from_label(
            labels[0], probability=raw_prediction.objects[0].score
        )

    @pytest.mark.parametrize("use_ellipse_shapes", [True, False])
    def test_rotated_rect_to_prediction_converter(
        self, use_ellipse_shapes, fxt_label_schema_factory
    ):
        # Arrange
        label_schema = fxt_label_schema_factory(Domain.ROTATED_DETECTION)
        labels = label_schema.get_labels(include_empty=False)
        coords = [1, 1, 4, 4]
        score = 0.51
        mask = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        raw_prediction = InstanceSegmentationResult(
            segmentedObjects=[
                SegmentedObject(*coords, mask=mask, score=score, id=1, str_label="")
            ],
            saliency_map=None,
            feature_vector=None,
        )
        height, width = mask.shape
        metadata = {"original_shape": (height, width, 3)}

        # Act
        converter = RotatedRectToPredictionConverter(
            label_schema, configuration={"use_ellipse_shapes": use_ellipse_shapes}
        )
        prediction = converter.convert_to_prediction(raw_prediction, metadata=metadata)

        # Assert
        assert converter.labels == labels
        assert len(prediction.annotations) == 1
        # raise Exception(prediction.annotations[0].labels[0], labels[0])
        # raise Exception(prediction.annotations[0].shape)
        if use_ellipse_shapes:
            assert prediction.annotations[0].shape == Ellipse(
                *coords_to_xmin_xmax_width_height(coords)
            )
        else:
            assert prediction.annotations[0].shape == RotatedRectangle.from_polygon(
                Polygon(
                    points=[
                        Point(x=1, y=1),
                        Point(x=3, y=1),
                        Point(x=3, y=3),
                        Point(x=1, y=3),
                    ]
                )
            )
        assert prediction.annotations[0].labels[0] == ScoredLabel.from_label(
            label=labels[0], probability=score
        )

    def test_segmentation_to_prediction_converter(self, fxt_segmentation_labels):
        # Arrange
        seg_labels = [fxt_segmentation_labels[0]]
        label_group = LabelGroup(
            labels=seg_labels, name="dummy segmentation label group"
        )
        label_schema = LabelSchema(label_groups=[label_group])
        labels = label_schema.get_labels(include_empty=False)
        result_image = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
            ]
        )
        soft_predictions = np.array(
            [
                [[0.9, 0.1, 0.1], [0.7, 0.1, 0.2], [0.9, 0.1, 0.1]],
                [[0.9, 0.0, 0.1], [0.9, 0.0, 0.1], [0.9, 0.0, 0.0]],
                [[0.2, 0.2, 0.6], [0.1, 0.2, 0.7], [0.2, 0.2, 0.6]],
            ]
        )
        raw_prediction = ImageResultWithSoftPrediction(
            resultImage=result_image,
            soft_prediction=soft_predictions,
            saliency_map=None,
            feature_vector=None,
        )

        # Act
        converter = SegmentationToPredictionConverter(label_schema)
        prediction = converter.convert_to_prediction(raw_prediction)

        # Assert
        assert converter.labels == labels
        assert len(prediction.annotations) == 1
        assert prediction.annotations[0].labels[0].name == labels[0].name
        assert prediction.annotations[0].shape == Polygon(
            points=[Point(1.0, 1.0), Point(0.0, 2.0), Point(1.0, 2.0), Point(2.0, 2.0)]
        )

    @pytest.mark.parametrize(
        "domain",
        [
            Domain.ANOMALY_CLASSIFICATION,
            Domain.ANOMALY_SEGMENTATION,
            Domain.ANOMALY_DETECTION,
        ],
    )
    def test_anomaly_to_prediction_converter(self, domain, fxt_label_schema_factory):
        # Arrange
        label_schema = fxt_label_schema_factory(domain)
        labels = label_schema.get_labels(include_empty=False)
        anomaly_map = np.ones((2, 2))
        pred_boxes = np.array([[2, 2, 4, 4]])
        pred_mask = np.ones((2, 2))
        raw_prediction = AnomalyResult(
            anomaly_map=anomaly_map,
            pred_boxes=pred_boxes,
            pred_mask=pred_mask,
            pred_label="Anomalous",
            pred_score=1.0,
        )

        # Act
        converter = AnomalyToPredictionConverter(label_schema)
        prediction = converter.convert_to_prediction(
            raw_prediction, image_shape=anomaly_map.shape
        )

        # Assert
        assert converter.labels == labels
        assert len(prediction.annotations) == 1
        assert prediction.annotations[0].labels[0] == ScoredLabel.from_label(
            next(label for label in labels if label.is_anomalous), probability=1.0
        )
        if domain == Domain.ANOMALY_SEGMENTATION:
            assert prediction.annotations[0].shape == Polygon(
                points=[
                    Point(0.0, 0.0),
                    Point(0.0, 1.0),
                    Point(1.0, 1.0),
                    Point(1.0, 0.0),
                ]
            )
        elif domain == Domain.ANOMALY_DETECTION:
            assert prediction.annotations[0].shape == Rectangle(
                *coords_to_xmin_xmax_width_height(pred_boxes[0])
            )
