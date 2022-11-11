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

from geti_sdk.data_models import Label, ScoredLabel


@pytest.fixture()
def fxt_label() -> Label:
    yield Label(
        name="Dog",
        color="#000000ff",
        group="Default classification group",
        is_empty=False,
    )


@pytest.fixture()
def fxt_scored_label(fxt_label: Label) -> ScoredLabel:
    yield ScoredLabel.from_label(label=fxt_label, probability=1.0)
