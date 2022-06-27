# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import ClassVar, Dict

import attr

from sc_api_tools.data_models.utils import str_to_media_type, attr_value_serializer


@attr.s(auto_attribs=True)
class MediaIdentifier:
    """
    Class representing media identification data as output by the SC /annotations
    REST endpoints

    :var type: Type of the media to which the annotation belongs
    """
    type: str = attr.ib(converter=str_to_media_type)

    def to_dict(self) -> Dict[str, str]:
        """
        Returns a dictionary form of the MediaIdentifier instance

        :return: Dictionary containing the media identifier data
        """
        return attr.asdict(self, value_serializer=attr_value_serializer)


@attr.s(auto_attribs=True)
class ImageIdentifier(MediaIdentifier):
    """
    Class representing image identification data used by the SC /annotations endpoints

    :var image_id: unique database ID of the image
    """
    _identifier_fields: ClassVar[str] = ["image_id"]

    image_id: str


@attr.s(auto_attribs=True)
class VideoFrameIdentifier(MediaIdentifier):
    """
    Class representing video frame identification data used by the SC /annotations
    endpoints

    :var frame_index: Index of the video frame in the full video
    :var video_id: unique database ID of the video to which the frame belongs
    """
    _identifier_fields: ClassVar[str] = ["video_id"]

    frame_index: int
    video_id: str


@attr.s(auto_attribs=True)
class VideoIdentifier(MediaIdentifier):
    """
    Class representing video identification data used by the SC /annotations endpoints

    :var video_id: unique database ID of the video
    """
    _identifier_fields: ClassVar[str] = ["video_id"]

    video_id: str
