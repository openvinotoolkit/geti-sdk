import json
import os
import time
import warnings
from typing import List, Union, Optional, Dict, Any, TypeVar

from requests import Response

from sc_api_tools.annotation_readers import AnnotationReader
from sc_api_tools.data_models import (
    Project,
    MediaList,
    Image,
    Video,
    VideoFrame,
    AnnotationScene, AnnotationKind
)
from sc_api_tools.http_session import SCSession
from sc_api_tools.rest_converters import AnnotationRESTConverter

AnnotationReaderType = TypeVar("AnnotationReaderType", bound=AnnotationReader)


class BaseAnnotationManager:
    """
    Class to up- or download annotations for 2d media to an existing project
    """

    def __init__(
            self,
            session: SCSession,
            project: Project,
            media_lists: List[Union[MediaList[Image], MediaList[Video]]],
            annotation_reader: Optional[AnnotationReaderType] = None
    ):
        self.session = session
        self.image_list: MediaList[Image] = MediaList([])
        self.video_list: MediaList[Video] = MediaList([])
        self.__set_media_lists(media_lists=media_lists)

        self.annotation_reader = annotation_reader
        self._project = project
        if annotation_reader is None:
            warnings.warn(
                "You did not specify an annotation reader for the annotation manager, "
                "this means it can only be used for annotation downloading, but not "
                "for uploading."
            )
            label_mapping = None
        else:
            label_mapping = self.__get_label_mapping(project)
        self._label_mapping = label_mapping

    def __set_media_lists(
            self, media_lists: List[Union[MediaList[Image], MediaList[Video]]]
    ):
        """
        Populates the image and video lists in the AnnotationManager.

        :param media_lists: Input media lists holding the images and videos in the
            project
        :return:
        """
        for media_list in media_lists:
            if len(media_list) > 0:
                if media_list.media_type == Image:
                    self.image_list.extend(media_list)
                elif media_list.media_type == Video:
                    self.video_list.extend(media_list)
                else:
                    raise ValueError(
                        f"Unsupported media type {media_list.media_type} found in "
                        f"media lists. Unable to process this media type in the "
                        f"AnnotationManager."
                    )

    def __get_label_mapping(self, project: Project) -> Dict[str, str]:
        """
        Get the mapping of the label names to the label ids for the project

        :param project:
        :return: Dictionary containing the label names as keys and the label ids as
            values
        """
        source_label_names = self.annotation_reader.get_all_label_names()
        project_label_mapping = project.pipeline.label_id_to_name_mapping
        project_label_name_to_id_mapping = {
            name: id_ for (id_, name) in project_label_mapping.items()
        }
        for source_label_name in source_label_names:
            if source_label_name not in project_label_name_to_id_mapping:
                raise ValueError(
                    f"Found label {source_label_name} in source labels, but this "
                    f"label is not in the project labels."
                )
        return project_label_name_to_id_mapping

    @property
    def label_mapping(self) -> Dict[str, str]:
        """
        Returns dictionary with label names as keys and label ids as values

        :return:
        """
        if self.annotation_reader is not None:
            if self._label_mapping is not None:
                return self._label_mapping
            else:
                self._label_mapping = self.__get_label_mapping(self._project)
        else:
            raise ValueError(
                "Unable to get label mapping for this annotation manager, no "
                "annotation reader has been defined."
            )

    def upload_annotation_for_2d_media_item(
            self, media_item: Union[Image, VideoFrame]
    ) -> Dict[str, Any]:
        """
        Uploads a new annotation for an image or video frame to the cluster. This will
        overwrite any current annotations for the media item.

        :param media_item: Image or VideoFrame to upload annotation for
        :return:
        """
        annotation_scene = self._read_2d_media_annotation_from_source(
            media_item=media_item
        )
        if annotation_scene.annotations:
            response = self.session.get_rest_response(
                url=f"{media_item.base_url}/annotations",
                method="POST",
                data=AnnotationRESTConverter.to_dict(
                    annotation_scene, deidentify=False
                )
            )
        else:
            response = {}
        return response

    @staticmethod
    def annotation_scene_from_rest_response(
            response_dict: Dict[str, Any]
    ) -> AnnotationScene:
        """
        Converts a dictionary with annotation data obtained from the SC /annotations
        rest endpoint into an annotation scene

        :param response_dict: Dictionary containing the annotation data
        :return: AnnotationScene object corresponding to the data in the response_dict
        """
        return AnnotationRESTConverter.from_dict(response_dict)

    def append_annotation_for_2d_media_item(
            self, media_item: Union[Image, VideoFrame]
    ) -> Union[Response, dict, list]:
        """
        Adds an annotation to the existing annotations for the `media_item`

        :param media_item: Image or VideoFrame to append the annotation for
        :return: Returns the response of the REST endpoint to post the updated
            annotation
        """
        new_annotation_scene = self._read_2d_media_annotation_from_source(
            media_item=media_item, preserve_shape_for_global_labels=True
        )
        try:
            annotation_scene = self.get_latest_annotation_for_2d_media_item(
                media_item
            )
        except ValueError:
            print(
                f"No existing annotation found for {str(media_item.type)} named "
                f"{media_item.name}"
            )
            annotation_scene = AnnotationScene(
                media_identifier=media_item.identifier,
                annotations=[],
                kind=AnnotationKind.ANNOTATION.value
            )
        annotation_scene.extend(new_annotation_scene.annotations)

        if annotation_scene.has_data:
            response = self.session.get_rest_response(
                url=f"{media_item.base_url}/annotations",
                method="POST",
                data=AnnotationRESTConverter.to_dict(annotation_scene, deidentify=False)
            )
        else:
            response = {}
        return response

    def get_latest_annotation_for_2d_media_item(
            self, media_item: Union[Image, VideoFrame]
    ) -> AnnotationScene:
        """
        Retrieve the latest annotation for an image or video frame from the cluster

        :param media_item: Image or VideoFrame to retrieve the annotations for
        :return: Dictionary containing the annotations data
        """
        response = self.session.get_rest_response(
            url=f"{media_item.base_url}/annotations/latest",
            method="GET"
        )
        return self.annotation_scene_from_rest_response(response)

    def _read_2d_media_annotation_from_source(
            self,
            media_item: Union[Image, VideoFrame],
            preserve_shape_for_global_labels: bool = False
    ) -> AnnotationScene:
        """
        Retrieve the annotation for the media_item, and return it in the
        proper format to be sent to the SC /annotations endpoint. This method uses the
        `self.annotation_reader` to get the annotation data.

        :param media_item: MediaItem to read the annotation for
        :return: Dictionary containing the annotation, in SC format
        """
        annotation_list = self.annotation_reader.get_data(
            filename=media_item.name,
            label_name_to_id_mapping=self.label_mapping,
            preserve_shape_for_global_labels=preserve_shape_for_global_labels
        )
        return AnnotationRESTConverter.from_dict(
            {
                "media_identifier": media_item.identifier,
                "annotations": annotation_list,
                "kind": AnnotationKind.ANNOTATION
            }
        )

    def _download_annotations_for_2d_media_list(
            self,
            media_list: Union[MediaList[Image], MediaList[VideoFrame]],
            path_to_folder: str,
            verbose: bool = True
    ) -> float:
        """
        Downloads annotations from the server to a target folder on disk

        :param media_list: List of images or video frames to download the annotations
            for
        :param path_to_folder: Folder to save the annotations to
        :return: Returns the time elapsed to download the annotations, in seconds
        """
        path_to_annotations_folder = os.path.join(path_to_folder, "annotations")
        if not os.path.exists(path_to_annotations_folder):
            os.makedirs(path_to_annotations_folder)
        if media_list.media_type == Image:
            media_name = 'image'
            media_name_plural = 'images'
        elif media_list.media_type == VideoFrame:
            media_name = 'video frame'
            media_name_plural = 'video frames'
        else:
            raise ValueError(
                "Invalid media type found in media_list, unable to download "
                "annotations."
            )
        if verbose:
            print(
                f"Starting annotation download... saving annotations for "
                f"{len(media_list)} {media_name_plural} to folder "
                f"{path_to_annotations_folder}"
            )
        t_start = time.time()
        download_count = 0
        skip_count = 0
        for media_item in media_list:
            try:
                annotation_scene = self.get_latest_annotation_for_2d_media_item(
                    media_item)
            except ValueError:
                if verbose:
                    print(
                        f"Unable to retrieve latest annotation for {media_name} "
                        f"{media_item.name}. Skipping this {media_name}"
                    )
                skip_count += 1
                continue
            kind = annotation_scene.kind
            if kind != AnnotationKind.ANNOTATION:
                if verbose:
                    print(
                        f"Received invalid annotation of kind {kind} for {media_name} "
                        f"with name{media_item.name}"
                    )
                skip_count += 1
                continue
            export_data = AnnotationRESTConverter.to_dict(annotation_scene)

            annotation_path = os.path.join(
                path_to_annotations_folder, media_item.name + '.json'
            )
            with open(annotation_path, 'w') as f:
                json.dump(export_data, f, indent=4)
            download_count += 1
        t_elapsed = time.time() - t_start
        if download_count > 0:
            msg = f"Downloaded {download_count} annotations to folder " \
                  f"{path_to_annotations_folder} in {t_elapsed:.1f} seconds."
        else:
            msg = f"No annotations were downloaded."
        if skip_count > 0:
            msg = msg + f" Was unable to retrieve annotations for {skip_count} " \
                        f"{media_name_plural}, these {media_name_plural} were skipped."
        if verbose:
            print(msg)
        return t_elapsed
