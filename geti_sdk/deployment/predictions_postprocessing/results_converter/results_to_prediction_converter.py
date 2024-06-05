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

"""Module implements the InferenceResultsToPredictionConverter class."""

import abc
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
from model_api.models.utils import (
    AnomalyResult,
    ClassificationResult,
    DetectionResult,
    ImageResultWithSoftPrediction,
    InstanceSegmentationResult,
)

from geti_sdk.data_models.annotations import Annotation
from geti_sdk.data_models.enums.domain import Domain
from geti_sdk.data_models.label import ScoredLabel
from geti_sdk.data_models.label_schema import LabelSchema
from geti_sdk.data_models.predictions import Prediction
from geti_sdk.data_models.shapes import (
    Ellipse,
    Point,
    Polygon,
    Rectangle,
    RotatedRectangle,
)
from geti_sdk.deployment.predictions_postprocessing.utils.detection_utils import (
    detection2array,
)
from geti_sdk.deployment.predictions_postprocessing.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
)


class InferenceResultsToPredictionConverter(metaclass=abc.ABCMeta):
    """Interface for the converter"""

    @abc.abstractmethod
    def convert_to_prediction(
        self, inference_results: NamedTuple, **kwargs
    ) -> Prediction:
        """
        Convert raw inference results to the Prediction format.

        :param inference_results: raw predictions from inference
        :return: Prediction object containing the shapes obtained from the raw predictions.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def convert_saliency_map(
        self, inference_results: NamedTuple, **kwargs
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract a saliency map from inference results and return in a unified format.

        :param inference_results: raw predictions from inference
        :return: Prediction object containing the shapes obtained from the raw predictions.
        """
        raise NotImplementedError


class ClassificationToPredictionConverter(InferenceResultsToPredictionConverter):
    """
    Converts ModelAPI Classification predictions to Prediction object.

    :param label_schema: LabelSchema containing the label info of the task
    """

    def __init__(self, label_schema: LabelSchema):
        all_labels = label_schema.get_labels(include_empty=True)
        # add empty labels if only one non-empty label exits
        non_empty_labels = [label for label in all_labels if not label.is_empty]
        self.labels = all_labels if len(non_empty_labels) == 1 else non_empty_labels
        # get the first empty label
        self.empty_label = next((label for label in all_labels if label.is_empty), None)
        multilabel = len(label_schema.get_groups(False)) > 1
        multilabel = multilabel and len(label_schema.get_groups(False)) == len(
            label_schema.get_labels(include_empty=False)
        )
        self.hierarchical = not multilabel and len(label_schema.get_groups(False)) > 1

        self.label_schema = label_schema

    def convert_to_prediction(
        self,
        inference_results: ClassificationResult,
        image_shape: Tuple[int, int, int],
        **kwargs,
    ) -> Prediction:  # noqa: ARG003
        """
        Convert ModelAPI ClassificationResult inference results to Prediction object.

        :param inference_results: classification labels represented in ModelAPI format (label_index, label_name, confidence)
        :param image_shape: shape of the input image
        :return: Prediction object with corresponding label
        """
        labels = []
        for label in inference_results.top_labels:
            labels.append(
                ScoredLabel.from_label(self.labels[label[0]], float(label[-1]))
            )

        if not labels and self.empty_label:
            labels = [ScoredLabel.from_label(self.empty_label, probability=1.0)]

        annotations = Annotation(
            shape=Rectangle.generate_full_box(image_shape[1], image_shape[0]),
            labels=labels,
        )
        return Prediction([annotations])

    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract a saliency map from inference results and return in a unified format.

        :param inference_results: classification labels represented in ModelAPI format (label_index, label_name, confidence)
        :param image_shape: shape of the input image
        :return: Prediction object with corresponding label
        """
        saliency_map = inference_results.saliency_map
        if len(saliency_map) == 0:
            return None
        saliency_map = cv2.resize(
            np.transpose(saliency_map.squeeze(0), (1, 2, 0)),
            dsize=(image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        if len(saliency_map.shape) == 2:
            saliency_map = np.expand_dims(saliency_map, axis=-1)
        saliency_map = np.transpose(saliency_map, (2, 0, 1))  # shape: (N classes, h, w)
        return {label.name: saliency_map[i] for i, label in enumerate(self.labels)}


class DetectionToPredictionConverter(InferenceResultsToPredictionConverter):
    """
    Converts ModelAPI Detection objects to Prediction object.

    :param label_schema: LabelSchema containing the label info of the task
    :param configuration: optional model configuration setting
    """

    def __init__(
        self, label_schema: LabelSchema, configuration: Optional[Dict[str, Any]] = None
    ):
        self.labels = label_schema.get_labels(include_empty=False)
        self.use_ellipse_shapes = False
        self.confidence_threshold = 0.0
        if configuration is not None:
            if "use_ellipse_shapes" in configuration:
                self.use_ellipse_shapes = configuration["use_ellipse_shapes"]
            if "confidence_threshold" in configuration:
                self.confidence_threshold = configuration["confidence_threshold"]

    def convert_to_prediction(
        self, inference_results: DetectionResult, **kwargs
    ) -> Prediction:
        """
        Convert ModelAPI DetectionResult inference results to Prediction object.

        :param inference_results: detection represented in ModelAPI format (label, confidence, x1, y1, x2, y2).

        _Note:
            - `label` can be any integer that can be mapped to `self.labels`
            - `confidence` should be a value between 0 and 1
            - `x1`, `x2`, `y1` and `y2` are expected to be in pixel
        :return: Prediction object containing the boxes obtained from the prediction
        """
        detections = detection2array(inference_results.objects)

        annotations = []
        if (
            len(detections)
            and detections.shape[1:] < (6,)
            or detections.shape[1:] > (7,)
        ):
            raise ValueError(
                f"Shape of prediction is not expected, expected (n, 7) or (n, 6) but got {detections.shape}"
            )

        for detection in detections:
            # Some OpenVINO models use an output shape of [7,]
            # If this is the case, skip the first value as it is not used
            _detection = detection[1:] if detection.shape == (7,) else detection

            label = int(_detection[0])
            confidence = _detection[1]
            scored_label = ScoredLabel.from_label(self.labels[label], confidence)
            coords = _detection[2:]
            shape: Union[Ellipse, Rectangle]

            if confidence < self.confidence_threshold:
                continue

            bbox_width = coords[2] - coords[0]
            bbox_height = coords[3] - coords[1]
            if self.use_ellipse_shapes:
                shape = Ellipse(coords[0], coords[1], bbox_width, bbox_height)
            else:
                shape = Rectangle(coords[0], coords[1], bbox_width, bbox_height)

            annotation = Annotation(shape=shape, labels=[scored_label])
            annotations.append(annotation)
        return Prediction(annotations)

    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract a saliency map from inference results and return in a unified format.

        :param inference_results: classification labels represented in ModelAPI format (label_index, label_name, confidence)
        :param image_shape: shape of the input image
        :return: Prediction object with corresponding label
        """
        saliency_map = inference_results.saliency_map
        if len(saliency_map) == 0:
            return None
        if isinstance(saliency_map, list):
            saliency_map = np.array(
                [
                    smap if len(smap) > 0 else np.zeros(image_shape[:2], dtype=np.uint8)
                    for smap in saliency_map
                ]
            )
        elif isinstance(saliency_map, np.ndarray):
            saliency_map = saliency_map.squeeze(0)
        else:
            raise ValueError(
                f"Unsupported saliency map type: {type(saliency_map)}. Expected list or numpy array."
            )
        saliency_map = cv2.resize(
            np.transpose(saliency_map, (1, 2, 0)),
            dsize=(image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        if len(saliency_map.shape) == 2:
            saliency_map = np.expand_dims(saliency_map, axis=-1)
        saliency_map = np.transpose(saliency_map, (2, 0, 1))  # shape: (N classes, h, w)
        return {label.name: saliency_map[i] for i, label in enumerate(self.labels)}


class RotatedRectToPredictionConverter(DetectionToPredictionConverter):
    """
    Converts ModelAPI Rotated Detection objects to Prediction.

    :param label_schema: LabelSchema containing the label info of the task
    """

    def convert_to_prediction(
        self, inference_results: InstanceSegmentationResult, **kwargs
    ) -> Prediction:
        """
        Convert ModelAPI instance segmentation inference results to a rotated bounding box annotation format.

        :param inference_results: segmentation represented in ModelAPI format
        :return: Prediction object containing the rotated boxes obtained from the segmentation contours
        :raises ValueError: if metadata is missing from the preprocess step
        """
        annotations = []
        shape: Union[RotatedRectangle, Ellipse]
        for obj in inference_results.segmentedObjects:
            if obj.score < self.confidence_threshold:
                continue
            if self.use_ellipse_shapes:
                shape = Ellipse(
                    obj.xmin, obj.ymin, obj.xmax - obj.xmin, obj.ymax - obj.ymin
                )
                annotations.append(
                    Annotation(
                        shape=shape,
                        labels=[
                            ScoredLabel.from_label(
                                self.labels[int(obj.id) - 1], float(obj.score)
                            )
                        ],
                    )
                )
            else:
                mask = obj.mask.astype(np.uint8)
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
                            x=point[0],
                            y=point[1],
                        )
                        for point in cv2.boxPoints(cv2.minAreaRect(contour))
                    ]
                    shape = Polygon(points=points)
                    annotations.append(
                        Annotation(
                            shape=RotatedRectangle.from_polygon(shape),
                            labels=[
                                ScoredLabel.from_label(
                                    self.labels[int(obj.id) - 1], float(obj.score)
                                )
                            ],
                        )
                    )
        return Prediction(annotations)


class MaskToAnnotationConverter(InferenceResultsToPredictionConverter):
    """Converts DetectionBox Predictions ModelAPI to Prediction object."""

    def __init__(
        self, label_schema: LabelSchema, configuration: Optional[Dict[str, Any]] = None
    ):
        self.labels = label_schema.get_labels(include_empty=False)
        self.use_ellipse_shapes = False
        self.confidence_threshold = 0.0
        if configuration is not None:
            if "use_ellipse_shapes" in configuration:
                self.use_ellipse_shapes = configuration["use_ellipse_shapes"]
            if "confidence_threshold" in configuration:
                self.confidence_threshold = configuration["confidence_threshold"]

    def convert_to_prediction(
        self, inference_results: Any, **kwargs: Dict[str, Any]
    ) -> Prediction:
        """
        Convert inference results to Prediction object.

        :param inference_results: Raw inference results from the model.
        :return: Prediction object.
        """
        annotations = []
        shape: Union[Polygon, Ellipse]
        for obj in inference_results.segmentedObjects:
            if obj.score < self.confidence_threshold:
                continue
            if self.use_ellipse_shapes:
                shape = shape = Ellipse(
                    obj.xmin, obj.ymin, obj.xmax - obj.xmin, obj.ymax - obj.ymin
                )
                annotations.append(
                    Annotation(
                        shape=shape,
                        labels=[
                            ScoredLabel.from_label(
                                self.labels[int(obj.id) - 1], float(obj.score)
                            )
                        ],
                    )
                )
            else:
                mask = obj.mask.astype(np.uint8)
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
                    contour = list(contour)
                    points = [
                        Point(
                            x=point[0][0],
                            y=point[0][1],
                        )
                        for point in contour
                    ]
                    shape = Polygon(points=points)
                    annotations.append(
                        Annotation(
                            shape=shape,
                            labels=[
                                ScoredLabel.from_label(
                                    self.labels[int(obj.id) - 1], float(obj.score)
                                )
                            ],
                        )
                    )
        return Prediction(annotations)

    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract a saliency map from inference results and return in a unified format.

        :param inference_results: classification labels represented in ModelAPI format (label_index, label_name, confidence)
        :param image_shape: shape of the input image
        :return: Prediction object with corresponding label
        """
        if len(inference_results.saliency_map) == 0:
            return None
        # Model API returns a list of np.ndarray for each label
        # Including `no_object` which is empty
        saliency_map = np.array(
            [
                smap if len(smap) > 0 else np.zeros(image_shape[:2], dtype=np.uint8)
                for smap in inference_results.saliency_map
            ]
        )  # shape: (N classes, h, w)
        return {label.name: saliency_map[i] for i, label in enumerate(self.labels)}


class SegmentationToPredictionConverter(InferenceResultsToPredictionConverter):
    """
    Converts ModelAPI Segmentation objects to Prediction object.

    :param label_schema: LabelSchema containing the label info of the task
    """

    def __init__(self, label_schema: LabelSchema):
        self.labels = label_schema.get_labels(include_empty=False)
        # NB: index=0 is reserved for the background label
        self.label_map = dict(enumerate(self.labels, 1))

    def convert_to_prediction(
        self, inference_results: ImageResultWithSoftPrediction, **kwargs  # noqa: ARG002
    ) -> Prediction:
        """
        Convert ModelAPI instance segmentation inference results to Prediction object.

        :param inference_results: segmentation represented in ModelAPI format
        :return: Prediction object containing the contour polygon obtained from the segmentation
        """
        annotations = create_annotation_from_segmentation_map(
            hard_prediction=inference_results.resultImage,
            soft_prediction=inference_results.soft_prediction,
            label_map=self.label_map,
        )
        return Prediction(annotations)

    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract a saliency map from inference results and return in a unified format.

        :param inference_results: classification labels represented in ModelAPI format (label_index, label_name, confidence)
        :param image_shape: shape of the input image
        :return: Prediction object with corresponding label
        """
        saliency_map = inference_results.saliency_map
        if len(saliency_map) == 0:
            return None
        saliency_map = np.transpose(saliency_map, (2, 0, 1))  # shape: (N classes, h, w)
        return {label.name: saliency_map[i + 1] for i, label in enumerate(self.labels)}


class AnomalyToPredictionConverter(InferenceResultsToPredictionConverter):
    """
    Convert ModelAPI AnomalyResult predictions to Prediction object.

    :param label_schema: LabelSchema containing the label info of the task
    """

    def __init__(self, label_schema: LabelSchema):
        self.labels = label_schema.get_labels(include_empty=False)
        self.normal_label = next(
            label for label in self.labels if not label.is_anomalous
        )
        self.anomalous_label = next(
            label for label in self.labels if label.is_anomalous
        )
        self.domain = self.anomalous_label.domain

    def convert_to_prediction(
        self, inference_results: AnomalyResult, image_shape: Tuple[int], **kwargs
    ) -> Prediction:  # noqa: ARG002
        """
        Convert ModelAPI AnomalyResult inferenceresults to sc_sdk annotations.

        :param inference_results: anomaly result represented in ModelAPI format (same for all anomaly tasks)
        :return: Prediction object based on the specific anomaly task:
            - Classification: single label (normal or anomalous).
            - Segmentation: contour polygon representing the segmentation.
            - Detection: predicted bounding boxes.
        """
        pred_label = inference_results.pred_label
        label = (
            self.anomalous_label
            if pred_label in ("Anomaly", "Anomalous")
            else self.normal_label
        )
        annotations: List[Annotation] = []
        if self.domain == Domain.ANOMALY_CLASSIFICATION:
            scored_label = ScoredLabel.from_label(
                label=label, probability=float(inference_results.pred_score)
            )
            annotations = [
                Annotation(
                    shape=Rectangle.generate_full_box(*image_shape[1::-1]),
                    labels=[scored_label],
                )
            ]
        elif self.domain == Domain.ANOMALY_SEGMENTATION:
            annotations = create_annotation_from_segmentation_map(
                hard_prediction=inference_results.pred_mask,
                soft_prediction=inference_results.anomaly_map.squeeze(),
                label_map={0: self.normal_label, 1: self.anomalous_label},
            )
        elif self.domain == Domain.ANOMALY_DETECTION:
            for box in inference_results.pred_boxes:
                annotations.append(
                    Annotation(
                        shape=Rectangle(
                            box[0], box[1], box[2] - box[0], box[3] - box[1]
                        ),
                        labels=[
                            ScoredLabel.from_label(
                                label=self.anomalous_label,
                                probability=inference_results.pred_score,
                            )
                        ],
                    )
                )
        else:
            raise ValueError(
                f"Cannot convert inference results for task '{self.domain.name}'. Only Anomaly tasks are supported."
            )
        if not annotations:
            scored_label = ScoredLabel.from_label(
                label=self.normal_label, probability=1.0
            )
            annotations = [
                Annotation(
                    labels=[scored_label],
                    shape=Rectangle.generate_full_box(*image_shape[1::-1]),
                )
            ]
        return Prediction(annotations)

    def convert_saliency_map(
        self,
        inference_results: NamedTuple,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract a saliency map from inference results and return in a unified format.

        :param inference_results: classification labels represented in ModelAPI format (label_index, label_name, confidence)
        :param image_shape: shape of the input image
        :return: Prediction object with corresponding label
        """
        # Normalizing Anomaly map
        saliency_map = inference_results.anomaly_map
        saliency_map -= saliency_map.min()
        saliency_map = saliency_map / (saliency_map.max() + 1e-12) * 255
        saliency_map = np.round(saliency_map).astype(np.uint8)  # shape: (h, w)
        return {self.anomalous_label.name: saliency_map}


class ConverterFactory:
    """
    Factory class for creating inference result to prediction converters based on the model's task.
    """

    @staticmethod
    def create_converter(
        label_schema: LabelSchema,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> InferenceResultsToPredictionConverter:
        """
        Create the appropriate inferencer object according to the model's task.

        :param label_schema: The label schema containing the label info of the task.
        :param configuration: Optional configuration for the converter. Defaults to None.
        :return: The created inference result to prediction converter.
        :raises ValueError: If the task type cannot be determined from the label schema.
        """
        domain = ConverterFactory._get_labels_domain(label_schema)

        if domain == Domain.CLASSIFICATION:
            return ClassificationToPredictionConverter(label_schema)
        if domain == Domain.DETECTION:
            return DetectionToPredictionConverter(label_schema, configuration)
        if domain == Domain.SEGMENTATION:
            return SegmentationToPredictionConverter(label_schema)
        if domain == Domain.ROTATED_DETECTION:
            return RotatedRectToPredictionConverter(label_schema, configuration)
        if domain == Domain.INSTANCE_SEGMENTATION:
            return MaskToAnnotationConverter(label_schema, configuration)
        if domain in (
            Domain.ANOMALY_CLASSIFICATION,
            Domain.ANOMALY_SEGMENTATION,
            Domain.ANOMALY_DETECTION,
        ):
            return AnomalyToPredictionConverter(label_schema)
        raise ValueError(f"Cannot create inferencer for task type '{domain.name}'.")

    @staticmethod
    def _get_labels_domain(label_schema: LabelSchema) -> Domain:
        """
        Return the domain (task type) associated with the model's labels.

        :param label_schema: The label schema containing the label info of the task.
        :return: The domain of the task.
        :raises ValueError: If the task type cannot be determined from the label schema.
        """
        try:
            return next(
                label.domain for label in label_schema.get_labels(include_empty=False)
            )
        except StopIteration:
            raise ValueError("Cannot determine the task for the model")
