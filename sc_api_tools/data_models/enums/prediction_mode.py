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


class PredictionMode(Enum):
    """
    This Enum represents the mode used to generate predictions in SC
    """
    LATEST = 'latest'
    AUTO = 'auto'
    ONLINE = 'online'

    def __str__(self):
        """
        Returns the string representation of the PredictionMode instance
        :return:
        """
        return self.value
