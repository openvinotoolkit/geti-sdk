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

from enum import Enum


class ShapeType(Enum):
    """
    Enum representing the types of shapes available for annotations on the Intel® Geti™
    platform.
    """

    RECTANGLE = "RECTANGLE"
    ELLIPSE = "ELLIPSE"
    POLYGON = "POLYGON"
    ROTATED_RECTANGLE = "ROTATED_RECTANGLE"
    KEYPOINT = "KEYPOINT"

    def __str__(self):
        """
        Return the string representation of the ShapeType instance.
        """
        return self.value
