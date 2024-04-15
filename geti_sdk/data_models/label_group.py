# Copyright (C) 2024 Intel Corporation
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
from typing import List, Optional

from geti_sdk.data_models.label import Label


class LabelGroupType(Enum):
    """Enum to indicate the LabelGroupType."""

    EXCLUSIVE = 1
    EMPTY_LABEL = 2


class LabelGroup:
    """
    Representation of a group of labels.
    """

    def __init__(
        self,
        name: str,
        labels: List[Label],
        group_type: LabelGroupType = LabelGroupType.EXCLUSIVE,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize a LabelGroup object.

        :param name: The name of the label group.
        :param labels: A list of Label objects associated with the group.
        :param group_type: The type of the label group. Defaults to LabelGroupType.EXCLUSIVE.
        :param id: The ID of the label group. Defaults to None.
        """
        self.id = id
        self.name = name
        self.group_type = group_type
        self.labels = sorted(labels, key=lambda label: label.id)
