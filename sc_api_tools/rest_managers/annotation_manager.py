import copy
import json
import os
import time
import warnings
from typing import Dict, Generic, TypeVar, Any, List, Optional, Union

from requests import Response

from sc_api_tools.annotation_readers.base_annotation_reader import AnnotationReader
from sc_api_tools.data_models import Project, MediaList, Image, Video, AnnotationScene, \
    MediaItem, VideoFrame
from sc_api_tools.http_session import SCSession
from sc_api_tools.rest_converters import AnnotationRESTConverter

AnnotationReaderType = TypeVar("AnnotationReaderType", bound=AnnotationReader)


class AnnotationManager(Generic[AnnotationReaderType]):
    """
    Class to upload annotations to an existing project
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
            label_mapping = self._get_label_mapping(project)
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
            if media_list.media_type == Image:
                self.image_list.extend(media_list)
            elif media_list.media_type == Video:
                self.video_list.extend(media_list)
            else:
                raise ValueError(
                    f"Unsupported media type {media_list.media_type} found in media "
                    f"lists. Unable to process this media type in the "
                    f"AnnotationManager."
                )

    def _get_label_mapping(self, project: Project) -> Dict[str, str]:
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
                self._label_mapping = self._get_label_mapping(self._project)
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
            media_item=media_item
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
                kind="annotation"
            )
        for annotation in annotation_scene.annotations:
            annotation.deidentify()
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
        return AnnotationRESTConverter.from_dict(response)

    def get_latest_annotations_for_video(self, video: Video) -> List[AnnotationScene]:
        """
        Retrieve all latest annotations for a video from the cluster

        :param video: Video to get the annotations for
        :return: List of AnnotationScene's, each entry corresponds to an
            AnnotationScene for a single frame in the video
        """
        response = self.session.get_rest_response(
            url=f"{video.base_url}/annotations/latest",
            method="GET"
        )
        return [
            AnnotationRESTConverter.from_dict(annotation_scene)
            for annotation_scene in response
        ]

    def _read_2d_media_annotation_from_source(
            self, media_item: Union[Image, VideoFrame]
    ) -> AnnotationScene:
        """
        Retrieve the annotation for the media_item, and return it in the
        proper format to be sent to the SC /annotations endpoint. This method uses the
        `self.annotation_reader` to get the annotation data.

        :param media_item: MediaItem to read the annotation for
        :return: Dictionary containing the annotation, in SC format
        """
        annotation_list = self.annotation_reader.get_data(
            filename=media_item.name, label_name_to_id_mapping=self.label_mapping
        )
        return AnnotationRESTConverter.from_dict(
            {
                "media_identifier": media_item.identifier,
                "annotations": annotation_list,
                "kind": "annotation"
            }
        )

    def upload_annotations_for_video(
            self, video: Video, append_annotations: bool = False
    ):
        """
        Uploads annotations for a video. If append_annotations is set to True,
        annotations will be appended to the existing annotations for the video in the
        project. If set to False, existing annotations will be overwritten.

        :param video: Video to upload annotations for
        :param append_annotations:
        :return:
        """
        annotation_filenames = self.annotation_reader.get_data_filenames()
        video_annotation_names = [
            filename for filename in annotation_filenames
            if filename.startswith(f"{video.name}_frame_")
        ]
        frame_indices = [int(name.split('_')[-1]) for name in video_annotation_names]
        video_frames = MediaList(
            [
                VideoFrame.from_video(video=video, frame_index=frame_index)
                for frame_index in frame_indices
            ]
        )
        upload_count = 0
        for frame in video_frames:
            if not append_annotations:
                response = self.upload_annotation_for_2d_media_item(media_item=frame)
            else:
                response = self.append_annotation_for_2d_media_item(media_item=frame)
            if response:
                upload_count += 1
        return upload_count

    def upload_annotations_for_videos(
            self, videos: MediaList[Video], append_annotations: bool = False
    ):
        """
        Uploads annotations for a list of videos. If append_annotations is set to True,
        annotations will be appended to the existing annotations for the video in the
        project. If set to False, existing annotations will be overwritten.

        :param videos: List of videos to upload annotations for
        :param append_annotations:
        :return:
        """
        print("Starting video annotation upload...")
        upload_count = 0
        for video in videos:
            upload_count += self.upload_annotations_for_video(
                video=video, append_annotations=append_annotations
            )
        if upload_count > 0:
            print(
                f"Upload complete. Uploaded {upload_count} new video frame annotations"
            )
        else:
            print(
                "No new video frame annotations were found."
            )

    def upload_annotations_for_images(
            self, images: MediaList[Image], append_annotations: bool = False
    ):
        """
        Uploads annotations for a list of images. If append_annotations is set to True,
        annotations will be appended to the existing annotations for the image in the
        project. If set to False, existing annotations will be overwritten.

        :param images: List of images to upload annotations for
        :param append_annotations:
        :return:
        """
        print("Starting image annotation upload...")
        upload_count = 0
        for image in images:
            if not append_annotations:
                response = self.upload_annotation_for_2d_media_item(media_item=image)
            else:
                response = self.append_annotation_for_2d_media_item(media_item=image)
            if response:
                upload_count += 1
        if upload_count > 0:
            print(f"Upload complete. Uploaded {upload_count} new image annotations")
        else:
            print(
                "No new image annotations were found."
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
            if kind != "annotation":
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
                json.dump(export_data, f)
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

    def download_annotations_for_video(
            self, video: Video, path_to_folder: str
    ) -> float:
        """
        Downloads video annotations from the server to a target folder on disk

        :param video: Video for which to download the annotations
        :param path_to_folder: Folder to save the annotations to
        :return: Returns the time elapsed to download the annotations, in seconds
        """
        annotations = self.get_latest_annotations_for_video(video=video)
        frame_list = MediaList[VideoFrame](
            [VideoFrame.from_video(
                video=video, frame_index=annotation.media_identifier.frame_index
            ) for annotation in annotations]
        )
        if len(frame_list) > 0:
            return self._download_annotations_for_2d_media_list(
                media_list=frame_list, path_to_folder=path_to_folder, verbose=False
            )
        else:
            return 0

    def download_annotations_for_images(
            self, images: MediaList[Image], path_to_folder: str
    ) -> float:
        """
        Downloads image annotations from the server to a target folder on disk

        :param images: List of images for which to download the annotations
        :param path_to_folder: Folder to save the annotations to
        :return: Returns the time elapsed to download the annotations, in seconds
        """
        return self._download_annotations_for_2d_media_list(
            media_list=images,
            path_to_folder=path_to_folder
        )

    def download_annotations_for_videos(
            self, videos: MediaList[Video], path_to_folder: str
    ) -> float:
        """
        Downloads annotations for a list of videos from the server to a target folder
        on disk

        :param videos: List of videos for which to download the annotations
        :param path_to_folder: Folder to save the annotations to
        :return: Time elapsed to download the annotations, in seconds
        """
        t_total = 0
        print(
            f"Starting annotation download... saving annotations for "
            f"{len(videos)} videos to folder {path_to_folder}/annotations"
        )
        for video in videos:
            t_total += self.download_annotations_for_video(
                video=video, path_to_folder=path_to_folder
            )
        print(f"Video annotation download finished in {t_total:.1f} seconds.")
        return t_total

    def download_all_annotations(self, path_to_folder: str) -> None:
        """
        Donwnloads all annotations for the project to a target folder on disk

        :param path_to_folder: Folder to save the annotations to
        """
        if len(self.image_list) > 0:
            self.download_annotations_for_images(
                images=self.image_list, path_to_folder=path_to_folder
            )
        if len(self.video_list) > 0:
            self.download_annotations_for_videos(
                self.video_list, path_to_folder=path_to_folder
            )

    def upload_annotations_for_all_media(self, append_annotations: bool = False):
        """
        Uploads annotations for all media in the project, If append_annotations is set
        to True, annotations will be appended to the existing annotations for the
        media on the server. If set to False, existing annotations will be overwritten.

        :param append_annotations: True to append annotations from the local disk to
            the existing annotations on the server, False to overwrite the server
            annotations by those on the local disk. Defaults to True
        """
        if len(self.image_list) > 0:
            self.upload_annotations_for_images(
                images=self.image_list, append_annotations=append_annotations
            )
        if len(self.video_list) > 0:
            self.upload_annotations_for_videos(
                videos=self.video_list, append_annotations=append_annotations
            )