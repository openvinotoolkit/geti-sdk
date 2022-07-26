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

import os
from importlib import resources

try:
    DEFAULT_DATA_PATH = str(resources.files("geti_sdk.demos") / "data")
except AttributeError:
    with resources.path("geti_sdk.demos", "data") as data_path:
        DEFAULT_DATA_PATH = str(data_path)

EXAMPLE_IMAGE_PATH = os.path.join(DEFAULT_DATA_PATH, "example", "dogs.png")
