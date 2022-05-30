from datetime import datetime
from typing import List

from openvino.model_zoo.model_api.models.utils import Detection

from sc_api_tools.data_models import Prediction, Label, Annotation, ScoredLabel
from sc_api_tools.data_models.shapes import Rectangle


def convert_detection_output(
        model_output: List[Detection],
        image_width: int,
        image_height: int,
        labels: List[Label]
) -> Prediction:
    """
    Convert the output of an OpenVINO inference model for a detection task to a
    Prediction object

    :param model_output: Nested list of Detections produced by a model for a detection
        task
    :param image_width: Width of the image (in pixels) to which the prediction applies
    :param image_height: Height of the image (in pixels) to which the prediction applies
    :param labels: List of labels belonging to the task
    :return: Prediction instance holding the prediction results
    """
    annotations: List[Annotation] = []
    empty_label = next((label for label in labels if label.is_empty), None)
    if len(model_output) == 0:
        if empty_label is not None:
            return Prediction(
                annotations=[
                    Annotation(
                        shape=Rectangle(0, 0, 1, 1),
                        labels=[ScoredLabel.from_label(empty_label, probability=1.0)]
                    )
                ]
            )
        else:
            raise ValueError(
                "Received empty model output, but no empty label was available to "
                "assign. Please include the empty label in the list of available "
                "labels."
            )
    if empty_label is not None:
        labels.pop(labels.index(empty_label))
    for detection in model_output:
        label_index = detection.id
        x_max = detection.xmax / image_width
        x_min = detection.xmin / image_width
        y_min = detection.ymin / image_height
        y_max = detection.ymax / image_height
        shape = Rectangle(
                x=x_min,
                y=y_min,
                height=y_max - y_min,
                width=x_max - x_min,
            )
        label = ScoredLabel.from_label(
            label=labels[label_index], probability=detection.score
        )
        annotations.append(Annotation(labels=[label], shape=shape))
    return Prediction(
        annotations=annotations,
        modified=datetime.now().isoformat()
    )
