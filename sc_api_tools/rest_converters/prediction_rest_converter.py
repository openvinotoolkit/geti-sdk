import copy
from typing import Dict, Any, List

from sc_api_tools.data_models import Prediction, Annotation
from sc_api_tools.data_models.predictions import ResultMedium
from sc_api_tools.rest_converters import AnnotationRESTConverter


class PredictionRESTConverter:
    @staticmethod
    def from_dict(prediction: Dict[str, Any]) -> Prediction:
        """
        Creates a Prediction object from a dictionary returned by the
        /predictions REST endpoint in SC

        :param prediction: dictionary representing a Prediction, which
            contains all prediction annotations for a certain media entity
        :return: Prediction object
        """
        input_copy = copy.deepcopy(prediction)
        annotations: List[Annotation] = []
        for annotation in prediction["annotations"]:
            if not isinstance(annotation, Annotation):
                annotations.append(
                    AnnotationRESTConverter.annotation_from_dict(annotation)
                )
            else:
                annotations.append(annotation)
        media_identifier = AnnotationRESTConverter._media_identifier_from_dict(
            prediction["media_identifier"]
        )

        result_media: List[ResultMedium] = []
        for result_medium in prediction["maps"]:
            if not isinstance(result_medium, ResultMedium):
                result_media.append(ResultMedium(**result_medium))
            else:
                result_media.append(result_medium)
        input_copy.update(
            {
                "annotations": annotations,
                "media_identifier": media_identifier,
                "maps": result_media
            }
        )
        return Prediction(**input_copy)
