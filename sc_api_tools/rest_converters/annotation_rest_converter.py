import copy

import attr
from omegaconf import OmegaConf
from sc_api_tools.data_models import MediaType, ScoredLabel, Image, \
    VideoFrame
from typing import List, Dict, Any, Union, cast

from sc_api_tools.data_models import AnnotationScene, Annotation
from sc_api_tools.data_models.annotations import (
    Shape,
    Ellipse,
    Polygon,
    Rectangle
)
from sc_api_tools.data_models.media_identifiers import MediaIdentifier, ImageIdentifier, \
    VideoFrameIdentifier
from sc_api_tools.data_models.enums import ShapeType
from sc_api_tools.data_models.utils import attr_enum_to_str, str_to_shape_type, \
    str_to_media_type
from sc_api_tools.utils.dictionary_helpers import remove_null_fields

SHAPE_TYPE_MAPPING = {
    ShapeType.ELLIPSE: Ellipse,
    ShapeType.RECTANGLE: Rectangle,
    ShapeType.POLYGON: Polygon
}
MEDIA_IDENTIFIER_MAPPING = {
    MediaType.IMAGE: ImageIdentifier,
    MediaType.VIDEO_FRAME: VideoFrameIdentifier
}


class AnnotationRESTConverter:
    @staticmethod
    def annotation_list_to_dict(
            media_item: Union[Image, VideoFrame],
            annotations: List[Union[Dict[str, Any], Annotation]]
    ):
        """
        Converts a list of annotations for a certain media_item to a dictionary that
        can be passed to the REST endpoint for annotation upload.

        :param media_item: MediaItem to which the annotations belong
        :param annotations: list of annotations
        :return:
        """
        converted_annotations: List[Dict[str, Any]] = []
        for annotation in annotations:
            if isinstance(annotation, Annotation):
                converted_annotations.append(
                    attr.asdict(
                        annotation, recurse=True, value_serializer=attr_enum_to_str
                    )
                )
            else:
                converted_annotations.append(annotation)
        return {
            "media_identifier": media_item.identifier.to_dict(),
            "annotations": converted_annotations,
        }

    @staticmethod
    def to_dict(
            annotation_scene: AnnotationScene, deidentify: bool = True
    ) -> Dict[str, Any]:
        """
        Converts an AnnotationScene to a dictionary. By default, removes any ID
        fields in the output dictionary

        :param annotation_scene: AnnotationScene object to convert
        :param deidentify: True to remove any unique database ID fields in the output,
            False to keep these fields. Defaults to True
        :return: Dictionary holding the serialized AnnotationScene data
        """
        if deidentify:
            annotation_scene.deidentify()
        annotation_dict = attr.asdict(
            annotation_scene, recurse=True, value_serializer=attr_enum_to_str
        )
        remove_null_fields(annotation_dict)
        return annotation_dict

    @staticmethod
    def _shape_from_dict(input_dict: Dict[str, Any]) -> Shape:
        """
        Convert a dictionary representing a shape to a Shape object

        :param input_dict:
        :return: Shape corresponding to the input dict
        """
        type_ = str_to_shape_type(input_dict.get("type"))
        class_type = SHAPE_TYPE_MAPPING[type_]
        return class_type(**input_dict)

    @staticmethod
    def _scored_label_from_dict(input_dict: Dict[str, Any]) -> ScoredLabel:
        """
        Creates a ScoredLabel object from an input dictionary

        :param input_dict:
        :return:
        """
        label_dict_config = OmegaConf.create(input_dict)
        schema = OmegaConf.structured(ScoredLabel)
        values = OmegaConf.merge(schema, label_dict_config)
        return cast(ScoredLabel, OmegaConf.to_object(values))

    @staticmethod
    def _annotation_from_dict(input_dict: Dict[str, Any]) -> Annotation:
        """
        Converst a dictionary representing an annotation to an Annotation object

        :param input_dict:
        :return:
        """
        input_copy = copy.deepcopy(input_dict)
        labels: List[ScoredLabel] = []
        for label in input_dict["labels"]:
            labels.append(AnnotationRESTConverter._scored_label_from_dict(label))
        shape = AnnotationRESTConverter._shape_from_dict(input_dict["shape"])
        input_copy.update({"labels": labels, "shape": shape})
        return Annotation(**input_copy)

    @staticmethod
    def _media_identifier_from_dict(input_dict: Dict[str, Any]) -> MediaIdentifier:
        """
        Converts a dictionary representing a media identifier to a MediaIdentifier
        object

        :param input_dict:
        :return:
        """
        if isinstance(input_dict, MediaIdentifier):
            return input_dict
        type_ = str_to_media_type(input_dict["type"])
        identifier_type = MEDIA_IDENTIFIER_MAPPING[type_]
        return identifier_type(**input_dict)

    @staticmethod
    def from_dict(annotation_scene: Dict[str, Any]) -> AnnotationScene:
        """
        Creates an AnnotationScene object from a dictionary returned by the
        /annotations REST endpoint in SC

        :param annotation_scene: dictionary representing an AnnotationScene, which
            contains all annotations for a certain media entity
        :return: AnnotationScene object
        """
        input_copy = copy.deepcopy(annotation_scene)
        annotations: List[Annotation] = []
        for annotation in annotation_scene["annotations"]:
            annotations.append(
                AnnotationRESTConverter._annotation_from_dict(annotation)
            )
        media_identifier = AnnotationRESTConverter._media_identifier_from_dict(
            annotation_scene["media_identifier"]
        )
        input_copy.update(
            {"annotations": annotations, "media_identifier": media_identifier}
        )
        return AnnotationScene(**input_copy)
