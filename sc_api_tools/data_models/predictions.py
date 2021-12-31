from typing import List, Optional, ClassVar, Dict

import cv2
import numpy as np

import attr
from sc_api_tools.data_models import AnnotationScene, AnnotationKind, Label
from sc_api_tools.data_models.media import MediaInformation
from sc_api_tools.data_models.shapes import Ellipse, Rectangle, Polygon
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
    type: Optional[str] = None
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
        # Label id is not defined for anomaly classification, it will always return
        # the saliency map for the 'Anomalous' label
        if self.name.lower() == 'anomaly map' and self.label_name is None:
            self.label_name = 'Anomalous'

    def get_data(self, session: SCSession) -> bytes:
        """
        Download the data belonging to this ResultMedium object

        :param session: REST session to the SC cluster from which this ResultMedium
            was generated
        :return: bytes object holding the data, if any is found
        """
        if self.data is None:
            if self.url is not None:
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

    @property
    def friendly_name(self) -> str:
        """
        Returns a human readable name with which the result medium can be identified

        :return:
        """
        return self.name + '_' + self.label_name


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

    @property
    def has_result_media(self) -> bool:
        """
        Returns True if this Prediction has result media belonging to it, False
        otherwise

        :return:
        """
        return len(self.maps) > 0

    def get_result_media_data(self, session: SCSession) -> List[ResultMedium]:
        """
        Downloads the data for all result media belonging to this prediction

        :param session: REST session to the SC cluster from which this Prediction
            was generated
        :return: List of result media, that have their data downloaded from the cluster
        """
        result: List[ResultMedium] = []
        for medium in self.maps:
            medium.get_data(session=session)
            result.append(medium)
        return result

    def as_mask(
            self,
            media_information: MediaInformation,
            probability_threshold: Optional[float] = None
    ):
        """
        Converts the shapes in the prediction to a mask that can be overlayed on an
        image

        :param media_information: MediaInformation object containing the width and
            heigth of the image for which the mask should be generated.
        :param probability_threshold: Threshold (between 0 and 1) for the probability.
            Shapes that are predicted with a probability below this threshold will
            not be plotted in the mask. If left as None (the default), all predictions
            will be shown.
        :return: np.ndarray holding the mask representation of the prediction
        """
        image_width = media_information.width
        image_heigth = media_information.height
        mask = np.zeros((image_heigth, image_width, 3))

        for annotation in self.annotations:
            max_prob_label_index = np.argmin(
                [label.probability for label in annotation.labels]
            )
            max_prob_label = annotation.labels[max_prob_label_index]
            if probability_threshold is not None:
                if max_prob_label.probability < probability_threshold:
                    continue
            color = max_prob_label.color_tuple
            line_thickness = 3
            shape = annotation.shape
            if isinstance(shape, (Ellipse, Rectangle)):
                x, y = int(shape.x * image_width), int(shape.y * image_heigth)
                width, height = int(shape.width * image_width), \
                                int(shape.height * image_heigth)
                if isinstance(shape, Ellipse):
                    cv2.ellipse(
                        mask,
                        center=(x, y),
                        axes=(width, height),
                        angle=0,
                        color=color,
                        startAngle=0,
                        endAngle=360,
                        thickness=line_thickness
                    )
                elif isinstance(shape, Rectangle):
                    if not shape.is_full_box:
                        cv2.rectangle(
                            mask,
                            pt1=(x, y),
                            pt2=(x+width, y+height),
                            color=color,
                            thickness=line_thickness
                        )
                    else:
                        origin = [
                            int(0.01*media_information.width),
                            int(0.99*media_information.height)
                        ]
                        for label in annotation.labels:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            cv2.putText(
                                mask,
                                label.name,
                                org=origin,
                                fontFace=font,
                                fontScale=font_scale,
                                color=label.color_tuple,
                                thickness=1
                            )
                            text_width, text_height = cv2.getTextSize(
                                label.name, font, font_scale, line_thickness
                            )[0]
                            origin[0] += text_width + 2
            elif isinstance(shape, Polygon):
                points = [
                    (int(x*image_width), int(y*image_heigth))
                    for (x, y) in shape.points_as_tuples()
                ]
                cv2.drawContours(
                    mask,
                    contours=points,
                    color=color,
                    thickness=line_thickness,
                    contourIdx=-1
                )
        return mask



