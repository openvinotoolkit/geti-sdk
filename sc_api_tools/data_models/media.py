import abc
import os
import io
from typing import Optional, List, Union, Dict, Any
import tempfile

import cv2
import numpy as np

import attr

from sc_api_tools.http_session import SCSession

from .enums import MediaType
from .media_identifiers import (
    MediaIdentifier,
    ImageIdentifier,
    VideoIdentifier,
    VideoFrameIdentifier
)
from .utils import str_to_media_type, str_to_datetime, numpy_from_buffer, \
    attr_value_serializer


@attr.s(auto_attribs=True)
class MediaInformation:
    """
    Class holding base information about a media item in SC

    :var display_url: URL that can be used to download the full size media entity
    :var height: Height of the media entity, in pixels
    :var width: Width of the media entity, in pixels
    """
    display_url: str
    height: int
    width: int


@attr.s(auto_attribs=True)
class VideoInformation(MediaInformation):
    """
    Class holding information about a video entity in SC

    :var duration: Duration of the video
    :var frame_count: Total number of frames in the video
    :var frame_stride: Frame stride of the video
    """
    duration: int
    frame_count: int
    frame_stride: int


@attr.s(auto_attribs=True)
class ImageInformation(MediaInformation):
    """
    Class holding information about an image entity in SC
    """
    pass


@attr.s(auto_attribs=True)
class VideoFrameInformation(MediaInformation):
    """
    Class holding information about a video frame in SC
    """
    frame_index: int
    video_id: str


@attr.s(auto_attribs=True)
class MediaItem:
    """
    Class representing a media entity in SC

    :var id: Unique database ID of the media entity
    :var name: Filename of the media entity
    :var state: Annotation state of the media entity
    :var type: MediaType of the entity
    :var upload_time: Time and date at which the entity was uploaded to the system
    :var thumbnail: URL that can be used to get a thumbnail for the media entity
    :var media_information: Container holding basic information such as width and
        height about the media entity
    """
    id: str
    name: str
    state: str
    type: str = attr.ib(converter=str_to_media_type)
    upload_time: str = attr.ib(converter=str_to_datetime)
    media_information: MediaInformation
    thumbnail: Optional[str] = None

    @property
    def download_url(self) -> str:
        """
        Returns the URL that can be used to download the full size media entity from SC

        :return: URL at which the media entity can be downloaded
        """
        return self.media_information.display_url.strip()

    @property
    def base_url(self) -> str:
        """
        Returns the base URL for the media item, which is the URL pointing to the
        media details of the entity

        :return: Base URL of the media entity
        """
        display_url_image = '/display/full'
        display_url_video = '/display/stream'
        if self.download_url.endswith(display_url_image):
            url = self.download_url[:-len(display_url_image)]
        elif self.download_url.endswith(display_url_video):
            url = self.download_url[:-len(display_url_video)]
        elif self.download_url.endswith(self.id):
            url = self.download_url
        else:
            raise ValueError(
                f"Unexpected end pattern found in display URL for media entity {self}. "
                f"Unable to construct base_url for entity"
            )
        return url

    @property
    def identifier(self) -> MediaIdentifier:
        """
        Returns the media identifier for the media item

        :return:
        """
        return MediaIdentifier(type=self.type)

    @abc.abstractmethod
    def get_data(self, session: SCSession) -> Union[np.ndarray, str]:
        """
        Get the pixel data for this MediaItem. Uses caching.

        For Image and VideoFrames, this method will return a np.ndarray. For Video
        objects, this method will return a path-like string, pointing to the video
        on the local disk.

        :param session: REST session to the SC cluster on which the MediaItem lives
        :raises ValueError: If the cache is empty and no data can be downloaded
            from the cluster
        :return: numpy array holding the pixel data for the MediaItem, or string
            pointing to the file location for Videos.
        """
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the MediaItem to a dictionary representation

        :return: Dictionary holding the annotation scene data
        """
        output_dict = attr.asdict(self, value_serializer=attr_value_serializer)
        return output_dict


@attr.s(auto_attribs=True)
class Image(MediaItem):
    """
    Class representing an image in SC

    :var media_information: Container holding basic information such as width and
            height about the image entity
    """
    media_information: ImageInformation = attr.ib(kw_only=True)

    def __attrs_post_init__(self):
        self._data: Optional[np.ndarray] = None

    @property
    def identifier(self) -> ImageIdentifier:
        """
        Returns the media identifier for the Image instance

        :return: ImageIdentifier object that contains the identifiers of the image
        """
        return ImageIdentifier(image_id=self.id, type=self.type)

    def get_data(self, session: SCSession) -> np.ndarray:
        """
        Get the pixel data for this Image. This method uses caching: If the cache is
        empty, it will download the data using the provided session. Otherwise it
        will return the cached data directly

        :param session: REST session to the SC cluster on which the Image lives
        :raises ValueError: If the cache is empty and no data can be downloaded
            from the cluster
        :return: Numpy.ndarray holding the pixel data for this Image.
        """
        if self._data is None:
            response = session.get_rest_response(
                url=self.download_url, method="GET", contenttype="jpeg"
            )
            if response.status_code == 200:
                self._data = numpy_from_buffer(response.content)
            else:
                raise ValueError(
                    f"Unable to retrieve data for image {self}, received "
                    f"response {response} from SC server."
                )
        return self._data

    @property
    def numpy(self) -> Optional[np.ndarray]:
        """
        Pixel data for the Image, as a numpy array of shape (width x heigth x 3).
        If this attribute is None, the pixel data should be downloaded from the
        cluster first using the `get_data` method

        :return: numpy.ndarray containing the pixel data
        """
        return self._data


@attr.s(auto_attribs=True)
class Video(MediaItem):
    """
    Class representing a video in SC

    :var media_information: Container holding basic information such as width,
            height and duration about the video entity
    """
    media_information: VideoInformation = attr.ib(kw_only=True)

    def __attrs_post_init__(self):
        self._data: Optional[str] = None
        self._needs_tempfile_deletion: bool = False

    @property
    def identifier(self) -> VideoIdentifier:
        """
        Returns the media identifier for the Video instance

        :return: VideoIdentifier object that contains the identifiers of the video
        """
        return VideoIdentifier(video_id=self.id, type=self.type)

    def get_data(self, session: SCSession) -> str:
        """
        Getting pixel data directly is not supported for Video entities, they have to
        be converted to VideoFrames first

        :param session:
        :return:
        """
        if self._data is None:
            response = session.get_rest_response(
                url=self.download_url, method="GET", contenttype="jpeg"
            )
            if response.status_code == 200:
                file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                file.write(response.content)
                file.close()
                self._data = file.name
                self._needs_tempfile_deletion = True
            else:
                raise ValueError(
                    f"Unable to retrieve data for image {self}, received "
                    f"response {response} from SC server."
                )
        return self._data

    def to_frames(
            self, frame_stride: Optional[int] = None, include_data: bool = False
    ) -> List['VideoFrame']:
        """
        Extract VideoFrames from the Video. Returns a list of VideoFrame objects.

        :param frame_stride: Stride to use for frame extraction. If left as None (the
            default), the frame_stride stored in the media_information is used.
        :param include_data: True to include pixel data for the frames, if available.
        :return: List of VideoFrames constructed from the Video
        """
        if frame_stride is None:
            frame_stride = self.media_information.frame_stride
        if frame_stride < 0:
            raise ValueError(
                f"Invalid frame stride {frame_stride}. Frame stride cannot be "
                f"negative."
            )
        elif frame_stride > self.media_information.frame_count:
            frame_stride = self.media_information.frame_count - 1
        frames_range = range(0, self.media_information.frame_count, frame_stride)
        if not include_data:
            frames = [
                VideoFrame.from_video(self, frame_index=index)
                for index in frames_range
            ]
        else:
            frames = [self.get_frame(frame_index=index) for index in frames_range]
        return frames

    def get_frame(self, frame_index: int) -> 'VideoFrame':
        """
        Returns a VideoFrame extracted from the Video, at the specified index.

        This method loads the numpy data for the frame, if available.

        :param frame_index: Index of the frame to extract
        :return: VideoFrame at the specified index
        """
        video_frame = VideoFrame.from_video(video=self, frame_index=frame_index)
        if self._data is not None:
            video_capture = cv2.VideoCapture(self._data)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = video_capture.read()
            if success:
                video_frame._data = frame
        return video_frame

    def __del__(self):
        """
        This method is called when the Video object is deleted. It cleans up the
        temporary file created to store the video data (if any)

        """
        if self._needs_tempfile_deletion:
            if os.path.exists(self._data):
                os.remove(self._data)


@attr.s(auto_attribs=True)
class VideoFrame(MediaItem):
    """
    Class representing a video frame in SC

    :var media_information: Container holding basic information such as width and
            height about the VideoFrame entity
    :var data: Pixel data for the VideoFrame. If this is None, the data can be
        downloaded using the 'get_data' method
    """
    media_information: VideoFrameInformation = attr.ib(kw_only=True)

    def __attrs_post_init__(self):
        self._data: Optional[np.ndarray] = None

    @classmethod
    def from_video(cls, video: Video, frame_index: int) -> 'VideoFrame':
        """
        Creates a VideoFrame entity from a `video` and a `frame_index`

        :param video: Video to extract the VideoFrame from
        :param frame_index: index at which the frame lives in the video
        :return:
        """
        base_url = f"{video.base_url}/frames/{frame_index}"
        frame_information = VideoFrameInformation(
            frame_index=frame_index,
            width=video.media_information.width,
            height=video.media_information.height,
            video_id=video.id,
            display_url=f"{base_url}/display/full"
        )
        return VideoFrame(
            name=f"{video.name}_frame_{frame_index}",
            type=str(MediaType.VIDEO_FRAME),
            upload_time=video.upload_time,
            thumbnail=f"{base_url}/display/thumb",
            media_information=frame_information,
            state=video.state,
            id=video.id
        )

    @property
    def identifier(self) -> VideoFrameIdentifier:
        """
        Returns the media identifier for the VideoFrame instance

        :return: VideoFrameIdentifier object that contains the identifiers of the
            video frame
        """
        return VideoFrameIdentifier(
            video_id=self.id,
            type=self.type,
            frame_index=self.media_information.frame_index
        )

    @property
    def numpy(self) -> Optional[np.ndarray]:
        """
        Pixel data for the Image, as a numpy array of shape (width x heigth x 3).
        If this attribute is None, the pixel data should be downloaded from the
        cluster first using the `get_data` method

        :return: numpy.ndarray containing the pixel data
        """
        return self._data

    def get_data(self, session: SCSession) -> np.ndarray:
        """
        Get the pixel data for this VideoFrame. This method uses caching: If the cache
        is empty, it will download the data using the provided session. Otherwise it
        will return the cached data directly

        :param session: REST session to the SC cluster on which the VideoFrame lives
        :raises ValueError: If the cache is empty and no data can be downloaded
            from the cluster
        :return: Numpy.ndarray holding the pixel data for this VideoFrame.
        """
        if self._data is None:
            response = session.get_rest_response(
                url=self.download_url, method="GET", contenttype="jpeg"
            )
            if response.status_code == 200:
                self._data = numpy_from_buffer(response.content)
            else:
                raise ValueError(
                    f"Unable to retrieve data for image {self}, received "
                    f"response {response} from SC server."
                )
        return self._data
