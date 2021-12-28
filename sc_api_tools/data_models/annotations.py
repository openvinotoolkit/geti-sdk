from typing import List, Union, Optional, ClassVar

import attr
from sc_api_tools.data_models import ScoredLabel
from sc_api_tools.data_models.media_identifiers import ImageIdentifier, \
    VideoFrameIdentifier
from sc_api_tools.data_models.utils import deidentify, \
    str_to_shape_type, str_to_datetime


@attr.s(auto_attribs=True)
class Shape:
    """
    Class representing a shape in SC

    :var type: Type of the shape
    """
    type: str = attr.ib(converter=str_to_shape_type)


@attr.s(auto_attribs=True)
class Rectangle(Shape):
    """
    Class representing a Rectangle in SC, as used in the /annotations REST endpoints

    NOTE: All coordinates and dimensions are relative to the full image

    :var x: X coordinate of the left side of the rectangle
    :var y: Y coordinate of the top of the rectangle
    :var width: Width of the rectangle
    :var height: Height of the rectangle
    """
    x: float
    y: float
    width: float
    height: float


@attr.s(auto_attribs=True)
class Ellipse(Shape):
    """
    Class representing an Ellipse in SC, as used in the /annotations REST endpoints

    NOTE: All coordinates and dimensions are relative to the full image

    :var x: Lowest x coordinate of the ellipse
    :var y: Lowest y coordinate of the ellipse
    :var width: Width of the ellipse
    :var height: Height of the ellipse
    """
    x: float
    y: float
    width: float
    height: float


@attr.s(auto_attribs=True)
class Point:
    """
    Class representing a point on a 2D coordinate system. Used to define Polygons in SC

    NOTE: All coordinates are defined relative to the full image size

    :var x: X coordinate of the point
    :var y: Y coordinate of the point
    """
    x: float
    y: float


@attr.s(auto_attribs=True)
class Polygon(Shape):
    """
    Class representing a polygon in SC, as used in the /annotations REST endpoints

    :var points: List of Points that make up the polygon
    """
    points: List[Point]


@attr.s(auto_attribs=True)
class Annotation:
    """
    Class representing a single annotation in SC.

    :var labels: List of labels belonging to the annotation
    :var modified: Date and time of the last modification made to this annotation
    :var shape: Shape of the annotation
    :var id: Unique database ID assigned to the annotation
    """
    _identifier_fields: ClassVar[str] = ["id", "modified"]

    labels: List[ScoredLabel]
    shape: Union[Rectangle, Ellipse, Polygon]
    modified: Optional[str] = attr.ib(converter=str_to_datetime, default=None)
    id: Optional[str] = None

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


@attr.s(auto_attribs=True)
class AnnotationScene:
    """
    Class representing all annotations for a certain media entity in SC

    :var annotations: List of annotations belonging to the media entity
    :var id: unique database ID of the AnnotationScene in SC
    :var kind: Kind of annotation (Annotation or Prediction)
    :var media_identifier: Identifier of the media entity to which this AnnotationScene
        applies
    :var modified: Data and time at which this AnnotationScene was last modified
    """
    _identifier_fields: ClassVar[str] = ["id", "modified"]

    annotations: List[Annotation]
    kind: str
    media_identifier: Optional[Union[ImageIdentifier, VideoFrameIdentifier]] = None
    id: Optional[str] = None
    modified: Optional[str] = attr.ib(converter=str_to_datetime, default=None)

    @property
    def has_data(self) -> bool:
        """
        Returns True if this AnnotationScene has annotation associated with it
        :return:
        """
        return len(self.annotations) > 0

    def deidentify(self):
        """
        Removes all unique database ID's from the annotationscene and the entities it
        contains

        :return:
        """
        deidentify(self)
        self.media_identifier = None
        for annotation in self.annotations:
            annotation.deidentify()

    def append(self, annotation: Annotation):
        """
        Add an annotation to the annotation scene

        :param annotation: Annotation to add
        :return:
        """
        self.annotations.append(annotation)

    def get_by_shape(self, shape: Shape) -> Optional[Annotation]:
        """
        Return the annotation belonging to a specific shape. Returns None if no
        Annotation is found for the shape.

        :param shape: Shape to return the annotation for
        :return:
        """
        return next(
            (
                annotation for annotation in self.annotations
                if annotation.shape == shape
            ), None
        )

    def extend(self, annotations: List[Annotation]):
        """
        Extend the list of annotations in the AnnotationScene with additional entries
        in the `annotations` list

        :param annotations: List of Annotations to add to the AnnotationScene
        :return:
        """
        for annotation in annotations:
            current_shape_annotation = self.get_by_shape(annotation.shape)
            if current_shape_annotation is not None:
                current_shape_annotation.extend_labels(annotation.labels)
            else:
                self.annotations.append(annotation)
