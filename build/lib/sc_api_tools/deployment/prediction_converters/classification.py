from typing import List, Tuple

from sc_api_tools.data_models import Label, Prediction, Annotation, ScoredLabel
from sc_api_tools.data_models.shapes import Rectangle


def convert_classification_output(
    model_output: List[List[Tuple[int, float]]],
    labels: List[Label]
) -> Prediction:
    """
    Convert the output of an OpenVINO inference model for a classification task to a
    Prediction object

    :param model_output: Nested list of classification results produced by a model for
        a classification task, for a single image
    :param labels: List of labels belonging to the task
    :return: Prediction instance holding the prediction results
    """
    shape = Rectangle(x=0, y=0, width=1, height=1)
    scored_labels: List[ScoredLabel] = []
    for label_index, score in model_output:
        label = labels[label_index]
        scored_labels.append(ScoredLabel.from_label(label=label, probability=score))
    return Prediction(
        annotations=[
            Annotation(shape=shape, labels=scored_labels)
        ]
    )
