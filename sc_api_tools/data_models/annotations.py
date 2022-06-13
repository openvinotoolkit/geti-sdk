from typing import List, Union, Optional, ClassVar

import attr

from ote_sdk.entities.annotation import Annotation as OteAnnotation

from sc_api_tools.data_models.label import ScoredLabel
from sc_api_tools.data_models.shapes import (
    Rectangle,
    Ellipse,
    Polygon, Shape, RotatedRectangle
)
from sc_api_tools.data_models.utils import (
    deidentify,
    str_to_datetime
)


@attr.s(auto_attribs=True)
class Annotation:
    """
    Class representing a single annotation in SC.

    :var labels: List of labels belonging to the annotation
    :var modified: Date and time of the last modification made to this annotation
    :var shape: Shape of the annotation
    :var id: Unique database ID assigned to the annotation
    :var labels_to_revisit: Optional list of database ID's of the labels that may not
        be up to date and therefore need to be revisited for this annotation
    """
    _identifier_fields: ClassVar[str] = ["id", "modified"]

    labels: List[ScoredLabel]
    shape: Union[Rectangle, Ellipse, Polygon, RotatedRectangle]
    modified: Optional[str] = attr.ib(converter=str_to_datetime, default=None)
    id: Optional[str] = None
    labels_to_revisit: Optional[List[str]] = None

    def deidentify(self):
        """
        Removes all unique database ID's from the annotation and the entities it
        contains

        :return:
        """
        deidentify(self)
        for label in self.labels:
            deidentify(label)

    @property
    def label_names(self) -> List[str]:
        """
        Returns a list of label names for this Annotation

        :return:
        """
        return [label.name for label in self.labels]

    def append_label(self, label: ScoredLabel):
        """
        Adds a label to the list of labels belonging to this annotation

        :param label:
        :return:
        """
        self.labels.append(label)

    def extend_labels(self, labels: List[ScoredLabel]):
        """
        Adds a list of labels to the labels already attached to this annotation

        :param labels: List of ScoredLabels to add
        :return:
        """
        self.labels.extend(labels)

    def pop_label_by_name(self, label_name: str):
        """
        Removes a label from the list of labels belonging to this annotation

        :param label_name: Name of the label to remove from the list
        :return:
        """
        index = None
        for index, label in enumerate(self.labels):
            if label.name == label_name:
                break
        if index is not None:
            self.labels.pop(index)

    @classmethod
    def from_ote(
            cls,
            ote_annotation: OteAnnotation,
            image_width: int,
            image_height: int
    ) -> 'Annotation':
        """
        Creates a :py:class:`~sc_api_tools.data_models.annotations.Annotation` instance
        from a given OTE SDK Annotation object.

        :param ote_annotation: OTE Annotation object to create the instance from
        :param image_width: Width of the image to which the annotation applies
        :param image_height: Height of the image to which the annotation applies
        :return: Annotation instance
        """
        shape = Shape.from_ote(
            ote_annotation.shape, image_width=image_width, image_height=image_height
        )
        labels = [
            ScoredLabel.from_ote(ote_label)
            for ote_label in ote_annotation.get_labels(include_empty=True)
        ]
        return Annotation(shape=shape, labels=labels, id=ote_annotation.id)
