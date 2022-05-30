from enum import Enum


class MediaType(Enum):
    """
    This enum represents the different media types in SC
    """
    IMAGE = 'image'
    VIDEO = 'video'
    VIDEO_FRAME = 'video_frame'

    def __str__(self) -> str:
        """
        Returns the string representation of the MediaType instance

        :return: string containing the media type
        """
        return self.value


SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mkv", ".mov"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"]
