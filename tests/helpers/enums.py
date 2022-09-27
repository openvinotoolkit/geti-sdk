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


class SdkTestMode(Enum):
    """
    This Enum represents the different modes available for running the tests. The
    available modes are:
        - ONLINE:  The tests are run against an actual Geti server. Real http
                     requests are being made but no cassettes are recorded
                     Use this mode to verify that the SDK data models are still up to
                     date with the current Geti REST contracts
        - OFFLINE: The tests are run using the recorded requests and responses. All
                     http requests are intercepted an no actual requests will be made.
                     Use this mode in a CI environment, or for fast testing of the SDK
                     logic
        - RECORD:  The tests are run against an actual Geti server. HTTP requests
                     are being made and all requests and responses are recorded to a new
                     set of cassettes for the SDK test suite. The old cassettes for the
                     tests are deleted.
    """

    ONLINE = "online"
    OFFLINE = "offline"
    RECORD = "record"
