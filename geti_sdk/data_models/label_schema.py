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

from typing import List, Optional

from geti_sdk.data_models.label import Label
from geti_sdk.data_models.label_group import LabelGroup, LabelGroupType


class LabelSchema:
    """
    The `LabelSchema` class defines the structure and properties of labels and label groups.

    :param label_groups: Optional list of `LabelGroup` objects representing the label groups in the schema
    """

    def __init__(self, label_groups: Optional[List[LabelGroup]] = None) -> None:
        """
        Initialize a new instance of the `LabelSchema` class.

        :param label_groups: Optional list of `LabelGroup` objects representing the label groups in the schema
        """
        self._groups = label_groups

    def get_labels(self, include_empty: bool = False) -> List[Label]:
        """
        Get the labels in the label schema.

        :param include_empty: Flag determining whether to include empty labels
        :return: List of all labels in the label schema
        """
        labels = {
            label
            for group in self._groups
            for label in group.labels
            if include_empty or not label.is_empty
        }
        return sorted(list(labels), key=lambda label: label.id)

    def get_groups(self, include_empty: bool = False) -> List[LabelGroup]:
        """
        Get the label groups in the label schema.

        :param include_empty: Flag determining whether to include empty label groups
        :return: List of all label groups in the label schema
        """
        if include_empty:
            return self._groups

        return [
            group
            for group in self._groups
            if group.group_type != LabelGroupType.EMPTY_LABEL
        ]
