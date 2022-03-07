from typing import List, Union, Optional, ClassVar

import attr

from sc_api_tools.data_models.label import ScoredLabel
from sc_api_tools.data_models.shapes import (
    Rectangle,
    Ellipse,
    Polygon
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
    shape: Union[Rectangle, Ellipse, Polygon]
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
