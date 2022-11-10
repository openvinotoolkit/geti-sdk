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

import pytest


@pytest.fixture()
def fxt_normalized_annotation_dict():
    yield {
        "labels": [{"probability": 1.0, "name": "Dog", "color": "#000000ff"}],
        "shape": {
            "x": 0.05,
            "y": 0.1,
            "width": 0.9,
            "height": 0.8,
            "type": "RECTANGLE",
        },
        "labels_to_revisit": [],
    }
