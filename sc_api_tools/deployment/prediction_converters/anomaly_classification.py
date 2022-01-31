from typing import List

import numpy as np

from sc_api_tools.data_models import Label, Prediction, ScoredLabel, Annotation
from sc_api_tools.data_models.shapes import Rectangle


def convert_anomaly_classification_output(
    model_output: np.ndarray,
    anomalous_label: Label,
    normal_label: Label
) -> Prediction:
    """
    Convert the output of an OpenVINO inference model for a classification task to a
    Prediction object

    :param model_output: Nested list of classification results produced by a model for
        a classification task, for a single image
    :param anomalous_label: Label that should be assigned to 'Anomalous' instances
    :param normal_label: Label that should be assigned to 'Normal' instances
    :return: Prediction instance holding the prediction results
    """
    shape = Rectangle(x=0, y=0, width=1, height=1)
    if model_output > 0.5:
        label = ScoredLabel.from_label(anomalous_label, probability=float(model_output))
    else:
        label = ScoredLabel.from_label(normal_label, probability=1-float(model_output))
    return Prediction(
        annotations=[
            Annotation(shape=shape, labels=[label])
        ]
    )
