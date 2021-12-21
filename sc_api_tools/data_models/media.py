from typing import Optional

import attr

from .enums import MediaType
from .media_identifiers import MediaIdentifier, ImageIdentifier, VideoIdentifier, \
    VideoFrameIdentifier
from .utils import str_to_media_type, str_to_datetime


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


@attr.s(auto_attribs=True)
class Image(MediaItem):
    """
    Class representing an image in SC

    :var media_information: Container holding basic information such as width and
            height about the image entity
    """
    media_information: ImageInformation = attr.ib(kw_only=True)

    @property
    def identifier(self) -> ImageIdentifier:
        """
        Returns the media identifier for the Image instance

        :return: ImageIdentifier object that contains the identifiers of the image
        """
        return ImageIdentifier(image_id=self.id, type=self.type)


@attr.s(auto_attribs=True)
class Video(MediaItem):
    """
    Class representing a video in SC

    :var media_information: Container holding basic information such as width,
            height and duration about the video entity
    """
    media_information: VideoInformation = attr.ib(kw_only=True)

    @property
    def identifier(self) -> VideoIdentifier:
        """
        Returns the media identifier for the Video instance

        :return: VideoIdentifier object that contains the identifiers of the video
        """
        return VideoIdentifier(video_id=self.id, type=self.type)


@attr.s(auto_attribs=True)
class VideoFrame(MediaItem):
    """
    Class representing a video frame in SC
    """
    media_information: VideoFrameInformation = attr.ib(kw_only=True)

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
