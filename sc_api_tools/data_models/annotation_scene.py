import copy
from pprint import pformat
from typing import ClassVar, List, Optional, Union, Dict, Any, Tuple

import attr
import cv2
import numpy as np

from ote_sdk.entities.annotation import (
    AnnotationSceneEntity,
    AnnotationSceneKind
)

from sc_api_tools.data_models.annotations import Annotation
from sc_api_tools.data_models.enums import AnnotationKind
from sc_api_tools.data_models.label import ScoredLabel
from sc_api_tools.data_models.media import MediaInformation
from sc_api_tools.data_models.media_identifiers import (
    ImageIdentifier,
    VideoFrameIdentifier
)
from sc_api_tools.data_models.shapes import (
    Shape,
    Ellipse,
    Rectangle,
    Polygon, RotatedRectangle,
)
from sc_api_tools.data_models.task_annotation_state import TaskAnnotationState
from sc_api_tools.data_models.utils import (
    str_to_annotation_kind,
    str_to_datetime,
    deidentify,
    attr_value_serializer,
    round_dictionary,
)
from sc_api_tools.utils import remove_null_fields


@attr.s(auto_attribs=True)
class AnnotationScene:
    """
    Class representing all annotations for a certain media entity in SC

    :var annotations: List of
        :py:class:`~sc_api_tools.data_models.annotations.Annotation`s belonging to the
        media entity
    :var id: unique database ID of the AnnotationScene in SC
    :var kind: :py:class:`~sc_api_tools.data_models.enums.annotation_kind.AnnotationKind`
        of the annotation (Annotation or Prediction)
    :var media_identifier: Identifier of the media entity to which this AnnotationScene
        applies
    :var modified: Date and time at which this AnnotationScene was last modified
    :var labels_to_revisit_full_scene: Optional list of database ID's of the labels
        that may not be up to date and therefore need to be revisited for this
        AnnotationScene.
    :var annotation_state_per_task: Optional dictionary holding the annotation state
        for this AnnotationScene for each task in the project pipeline.
    """
    _identifier_fields: ClassVar[List[str]] = ["id", "modified"]
    _GET_only_fields: ClassVar[List[str]] = ["annotation_state_per_task"]

    annotations: List[Annotation]
    kind: str = attr.ib(
        converter=str_to_annotation_kind, default=AnnotationKind.ANNOTATION
    )
    media_identifier: Optional[Union[ImageIdentifier, VideoFrameIdentifier]] = None
    id: Optional[str] = None
    modified: Optional[str] = attr.ib(converter=str_to_datetime, default=None)
    labels_to_revisit_full_scene: Optional[List[str]] = None
    annotation_state_per_task: Optional[List[TaskAnnotationState]] = None

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

    def prepare_for_post(self):
        """
        Removes all fields that are not valid for making a POST request to the
        /annotations endpoint

        :return:
        """
        for field_name in self._GET_only_fields:
            setattr(self, field_name, None)

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the AnnotationScene to a dictionary representation

        :return: Dictionary holding the annotation scene data
        """
        output_dict = attr.asdict(self, value_serializer=attr_value_serializer)
        return output_dict

    @property
    def overview(self) -> str:
        """
        Returns a string that gives an overview of the annotation scene

        :return: overview string of the annotation scene
        """
        deidentified = copy.deepcopy(self)
        deidentified.deidentify()
        dict_output = deidentified.to_dict()
        for annotation_dict in dict_output["annotations"]:
            shape_dict = annotation_dict["shape"]
            annotation_dict["shape"] = round_dictionary(shape_dict)
        remove_null_fields(dict_output)
        return pformat(dict_output)

    @staticmethod
    def _add_shape_to_mask(
            shape: Shape,
            mask: np.ndarray,
            labels: List[ScoredLabel],
            color: Tuple[int, int, int],
            line_thickness: int
    ) -> np.ndarray:
        """
        Draws an SC shape entity `shape` on the pixel level mask `mask`. The shape
        will be drawn in the color specified as R,G,B tuple in `color`, using a line
        thickness `line_thickness` (in pixels).

        If the shape represents classification annotations or predictions, this method
        will print the name of the classification labels in the lower-left corner of
        the mask

        :param shape: Shape to add
        :param mask: Mask to draw the shape on
        :param labels: List of labels belonging to the shape
        :param color: RGB tuple representing the color in which the shape should be
            drawn
        :param line_thickness: Line thickness (in pixels) with which the shape should
            be drawn
        :return: Mask with the shape drawn on it
        """
        image_height, image_width = mask.shape[0:-1]
        if isinstance(shape, (Ellipse, Rectangle)):
            x, y = int(shape.x * image_width), int(shape.y * image_height)
            width = int(shape.width * image_width)
            height = int(shape.height * image_height)
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
                cv2.ellipse(
                    mask,
                    center=(x, y),
                    axes=(width, height),
                    angle=0,
                    color=(1, 1, 1),
                    startAngle=0,
                    endAngle=360,
                    thickness=1
                )
            elif isinstance(shape, Rectangle):
                if not shape.is_full_box:
                    cv2.rectangle(
                        mask,
                        pt1=(x, y),
                        pt2=(x + width, y + height),
                        color=color,
                        thickness=line_thickness
                    )
                    cv2.rectangle(
                        mask,
                        pt1=(x, y),
                        pt2=(x + width, y + height),
                        color=(1, 1, 1),
                        thickness=1
                    )
                else:
                    origin = [
                        int(0.01 * image_width),
                        int(0.99 * image_height)
                    ]
                    for label in labels:
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
        elif isinstance(shape, RotatedRectangle):
            shape = shape.to_polygon()
        if isinstance(shape, Polygon):
            contour = shape.points_as_contour(
                image_width=image_width, image_height=image_height
            )
            cv2.drawContours(
                mask,
                contours=[contour],
                color=color,
                thickness=line_thickness,
                contourIdx=-1
            )
            cv2.drawContours(
                mask,
                contours=[contour],
                color=(1, 1, 1),
                thickness=1,
                contourIdx=-1
            )
        return mask

    def as_mask(self, media_information: MediaInformation) -> np.ndarray:
        """
        Converts the shapes in the annotation scene to a mask that can be overlayed on
        an image

        :param media_information: MediaInformation object containing the width and
            height of the image for which the mask should be generated.
        :return: np.ndarray holding the mask representation of the annotation scene
        """
        image_width = media_information.width
        image_height = media_information.height
        mask = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        for annotation in self.annotations:
            label_to_copy_color = annotation.labels[0]
            line_thickness = 3
            shape = annotation.shape
            mask = self._add_shape_to_mask(
                shape=shape,
                mask=mask,
                labels=annotation.labels,
                color=label_to_copy_color.color_tuple,
                line_thickness=line_thickness
            )
        return mask

    def apply_identifier(
            self, media_identifier: Union[ImageIdentifier, VideoFrameIdentifier]
    ) -> 'AnnotationScene':
        """
        Applies a `media_identifier` to the current AnnotationScene instance, such
        that the SC cluster will recognize this AnnotationScene as belonging to the
        media item identified by the media_identifier.

        This method creates and returns a new AnnotationScene instance. The instance
        on which this method is called remains unmodified.

        :param media_identifier: Image or VideoFrame identifier to apply
        :return: new AnnotationScene instance with the identifiers set according to
            `media_identifier`
        """
        new_annotation = copy.deepcopy(self)
        new_annotation.media_identifier = media_identifier
        new_annotation.id = ""
        new_annotation.modified = None
        for annotation in new_annotation.annotations:
            annotation.id = ""
            annotation.modified = ""
        return new_annotation

    @classmethod
    def from_ote(cls, ote_annotation_scene: AnnotationSceneEntity) -> 'AnnotationScene':
        """
        Creates a :py:class:`~sc_api_tools.data_models.annotation_scene.AnnotationScene`
        instance from a given OTE SDK AnnotationSceneEntity object.

        :param ote_annotation_scene: OTE AnnotationSceneEntity object to create the
            instance from
        :return: AnnotationScene instance
        """
        annotations = [
            Annotation.from_ote(annotation)
            for annotation in ote_annotation_scene.annotations
        ]
        return cls(
            annotations=annotations,
            id=ote_annotation_scene.id,
        )
