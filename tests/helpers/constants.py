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

BASE_TEST_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMMY_HOST = "dummy_host"
CASSETTE_BASE_PATH = os.path.join(BASE_TEST_PATH, "fixtures", "cassettes")
CASSETTE_PATH_KEY = "VERSIONED_CASSETTES_PATH"
RECORD_CASSETTE_KEY = "RECORD_CASSETTES_TEMPORARY_DIR"
CASSETTE_EXTENSION = "cassette"
PROJECT_PREFIX = "geti_sdk_test"
DUMMY_USER = "dummy_user"
DUMMY_PASSWORD = "dummy_password"  # nosec B105
DUMMY_TOKEN = "dummy_token"  # nosec B105
DUMMY_WORKSPACE_ID = "dummy_workspace_id"
DUMMY_ORGANIZATION_ID = "dummy_organization_id"
