from typing import Union, Optional, List

from sc_api_tools.data_models import (
    Project,
    Image,
    VideoFrame,
    MediaItem,
    Video,
    Prediction
)
from sc_api_tools.data_models.enums import PredictionType
from sc_api_tools.http_session import SCSession

from sc_api_tools.rest_converters import PredictionRESTConverter


class PredictionManager:
    """
    Class to download predictions from an existing SC project
    """

    def __init__(self, session: SCSession, project: Project):
        self.session = session
        self.project = project
        self._labels = project.get_all_labels()

    def _get_prediction_for_media_item(
            self,
            media_item: MediaItem,
            prediction_type: Union[str, PredictionType] = PredictionType.LATEST
    ) -> Optional[Union[Prediction, List[Prediction]]]:
        """
        Gets the prediction for a media item

        :param media_item: Image or VideoFrame to get the prediction for
        :return: Prediction
        """
        if isinstance(prediction_type, str):
            prediction_type = PredictionType(prediction_type)
        try:
            response = self.session.get_rest_response(
                url=f"{media_item.base_url}/predictions/{str(prediction_type)}",
                method="GET"
            )
            if isinstance(media_item, (Image, VideoFrame)):
                result = PredictionRESTConverter.from_dict(response)
                result.resolve_labels_for_result_media(
                    labels=self._labels
                )
            elif isinstance(media_item, Video):
                result = [
                    PredictionRESTConverter.from_dict(
                        prediction
                    ).resolve_labels_for_result_media(labels=self._labels)
                    for prediction in response
                ]
            else:
                raise TypeError(
                    f"Getting predictions is not supported for media item of type "
                    f"{media_item.type}. Unable to retrieve predictions."
                )
        except ValueError as error:
            msg = f"Unable to retrieve prediction for {media_item.type}."
            if error.args[-1] == 204:
                msg += f" The prediction for the {media_item.type} with name " \
                       f"'{media_item.name}' is not available in project " \
                       f"'{self.project.name}'. Has a model been trained for the " \
                       f"project?"
            else:
                msg += f" Server responded with error message: {str(error)}"
            print(msg)
            result = None
        return result

    def get_image_prediction(self, image: Image) -> Prediction:
        """
        Gets a prediction for an image from the SC cluster

        :param image: Image to get the prediction for. The image has to be present in
            the project on the cluster already.
        :return: Prediction for the image
        """
        return self._get_prediction_for_media_item(
            media_item=image, prediction_type=PredictionType.ONLINE
        )
