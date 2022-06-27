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

from typing import Optional

import attr


@attr.define()
class Performance:
    """
    Class holding the performance metrics for a project or model in SC

    :var score: Overall score of the project or model
    :var local_score: Accuracy of the model or project with respect to object
        localization
    :var global_score: Accuracy of the model or project with respect to global
        classification of the full image
    """
    score: Optional[float] = None
    local_score: Optional[float] = None
    global_score: Optional[float] = None
