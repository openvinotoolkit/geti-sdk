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

from typing import List, Dict

import pytest


@pytest.fixture()
def fxt_hierarchical_classification_labels() -> List[Dict[str, str]]:
    yield [
            {"name": "animal"},
            {"name": "dog", "parent_id": "animal"},
            {"name": "cat", "parent_id": "animal"},
            {"name": "vehicle"},
            {"name": "car", "parent_id": "vehicle", "group": "vehicle type"},
            {"name": "taxi", "parent_id": "vehicle", "group": "vehicle type"},
            {"name": "truck", "parent_id": "vehicle", "group": "vehicle type"},
            {"name": "red", "parent_id": "vehicle", "group": "vehicle color"},
            {"name": "blue", "parent_id": "vehicle", "group": "vehicle color"},
            {"name": "black", "parent_id": "vehicle", "group": "vehicle color"},
            {"name": "grey", "parent_id": "vehicle", "group": "vehicle color"}
    ]


@pytest.fixture()
def fxt_default_labels() -> List[str]:
    yield ["cube", "cylinder"]
