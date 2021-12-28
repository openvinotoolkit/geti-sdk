from typing import List, Optional, ClassVar

import attr
from sc_api_tools.data_models import AnnotationScene, AnnotationKind, Label
from sc_api_tools.data_models.utils import str_to_annotation_kind, deidentify
from sc_api_tools.http_session import SCSession


@attr.s(auto_attribs=True)
class ResultMedium:
    """
    Class representing a single result medium in SC.

    :var name: Name of the result medium option
    :var type: Type of the result medium represented by this object
    :var url: URL at which the full result medium can be downloaded
    :var id: Unique database ID assigned to the result medium
    :var label_id: Unique database ID of the label referenced by this result medium
    """

    _identifier_fields: ClassVar[str] = ["id", "data", "label_id", "url"]

    name: str
    type: str
    url: Optional[str] = None
    label_id: Optional[str] = None
    id: Optional[str] = None
    label_name: Optional[str] = attr.ib(init=False, default=None)
    data: Optional[bytes] = attr.ib(default=None, repr=False, init=False)

    def resolve_label_name(self, labels: List[Label]):
        """
        Add the label name to the result medium, by matching the label_id to a list of
        Labels

        :param labels: List of Labels to get the name from
        """
        self.label_name = next(
            (label.name for label in labels if label.id == self.label_id), None
        )

    def get_data(self, session: SCSession) -> Optional[bytes]:
        """
        Download the data belonging to this ResultMedium object

        :param session: REST session to the SC cluster from which this ResultMedium
            was generated
        :return: bytes object holding the data, if any is found
        """
        if self.data is None:
            response = session.get_rest_response(
                url=self.url, method="GET", contenttype="jpeg"
            )
            if response.status_code == 200:
                self.data = response.content
            else:
                raise ValueError(
                    f"Unable to retrieve data for result medium {self}, received "
                    f"response {response} from SC server."
                )
        return self.data


@attr.s(auto_attribs=True)
class Prediction(AnnotationScene):
    """
    Class representing the predictions for a certain media entity in SC

    :var annotations: List of predictions belonging to the media entity
    :var id: unique database ID of the Prediction in SC
    :var kind: Kind of prediction (Annotation or Prediction)
    :var media_identifier: Identifier of the media entity to which this Prediction
        applies
    :var modified: Data and time at which this Prediction was last modified
    :var maps: List of additional result media belonging to this prediction
    """
    kind: str = attr.ib(
        converter=str_to_annotation_kind,
        default=AnnotationKind.PREDICTION,
        kw_only=True
    )
    maps: List[ResultMedium] = attr.ib(factory=list, kw_only=True)

    def resolve_labels_for_result_media(self, labels: List[Label]):
        """
        Resolve the label names for all result media available with this Prediction

        :param labels: List of Labels for the project, from which the names are taken
        """
        for map_ in self.maps:
            map_.resolve_label_name(labels=labels)

    def deidentify(self):
        """
        Removes all unique database ID's from the prediction and the entities it
        contains

        :return:
        """
        deidentify(self)
        self.media_identifier = None
        for annotation in self.annotations:
            annotation.deidentify()
        for map_ in self.maps:
            deidentify(map_)
