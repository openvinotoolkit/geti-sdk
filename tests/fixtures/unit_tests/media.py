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
from typing import Dict

import pytest

from geti_sdk.data_models import MediaType
from geti_sdk.data_models.media_identifiers import ImageIdentifier


@pytest.fixture()
def fxt_image_identifier() -> ImageIdentifier:
    yield ImageIdentifier(image_id="image_0", type=MediaType.IMAGE)


@pytest.fixture()
def fxt_image_identifier_rest() -> Dict[str, str]:
    yield {"image_id": "image_0", "type": "image"}
