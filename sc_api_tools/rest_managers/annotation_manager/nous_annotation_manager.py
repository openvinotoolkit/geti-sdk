import re
from typing import Union

from sc_api_tools.data_models import (
    Image,
    Video,
    VideoFrame,
    AnnotationScene,
    AnnotationKind
)
from sc_api_tools.data_models.containers import MediaList
from sc_api_tools.rest_clients import AnnotationClient
from sc_api_tools.rest_converters import AnnotationRESTConverter
from sc_api_tools.annotation_readers.nous_annotation_reader import NOUSAnnotationReader



class NOUSAnnotationManager(AnnotationClient[NOUSAnnotationReader]):
    """
    Class to up- or download annotations for images or videos to an existing project.
    The purpose of this class is to handle annotation upload for annotations in
    exported NOUS projects
    """

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
        video_fname = video.name.rsplit('_', 1)[0]
        video_annotation_names = [
            filename for filename in annotation_filenames
            if filename.startswith(f"{video_fname}_frame_")
        ]
        frame_pattern = re.compile("_frame_[0-9]+_")
        frame_indices = [
            int(frame_pattern.search(name).group().strip('_').split("_")[-1])
            for name in video_annotation_names
            if frame_pattern.search(name) is not None
        ]
        video_frames = MediaList(
            [
                VideoFrame.from_video(video=video, frame_index=frame_index)
                for frame_index in frame_indices
            ]
        )

        for frame in video_frames:
            frame.parent_video_file = video.name

        upload_count = 0
        for frame in video_frames:
            if not append_annotations:
                response = self._upload_annotation_for_2d_media_item(media_item=frame)
            else:
                response = self._append_annotation_for_2d_media_item(media_item=frame)
            if response.has_data:
                upload_count += 1
        return upload_count

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
        if hasattr(media_item, 'parent_video_file'):
            filename = media_item.parent_video_file
        else:
            filename = media_item.name

        if hasattr(media_item.media_information, 'frame_index'):
            frame = media_item.media_information.frame_index
        else:
            frame = -1

        # print(self.label_mapping)
        annotation_list = self.annotation_reader.get_data(
            filename=filename,
            label_name_to_id_mapping=self.label_mapping,
            preserve_shape_for_global_labels=preserve_shape_for_global_labels,
            frame=frame,
            media_item=media_item
        )
        return AnnotationRESTConverter.from_dict(
            {
                "media_identifier": media_item.identifier,
                "annotations": annotation_list,
                "kind": AnnotationKind.ANNOTATION
            }
        )
