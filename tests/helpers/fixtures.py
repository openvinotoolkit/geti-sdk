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
from typing import List

from .constants import BASE_TEST_PATH


def get_sdk_fixtures() -> List[str]:
    """
    Returns the list of fixtures available to the SDK

    :return: list of fixture paths for pytest to import
    """
    fixture_filenames = os.listdir(os.path.join(BASE_TEST_PATH, "fixtures"))
    fixtures: List[str] = []
    for filename in fixture_filenames:
        if filename.endswith(".py") and not filename.startswith("__"):
            fixtures.append(f"tests.fixtures.{filename[0:-3]}")
    return fixtures
