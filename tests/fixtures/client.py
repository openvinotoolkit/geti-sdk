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

from sc_api_tools import SCRESTClient
from sc_api_tools.http_session import ClusterConfig, SCSession
from tests.helpers.constants import CASSETTE_EXTENSION

DUMMY_USER = "dummy_user"
DUMMY_PASSWORD = "dummy_password"


@pytest.fixture(scope="module")
def fxt_sc_session(fxt_vcr, fxt_server_config: ClusterConfig) -> SCSession:
    """
    This fixture returns an SCSession instance which has already performed
    authentication
    """
    with fxt_vcr.use_cassette(
        f"session.{CASSETTE_EXTENSION}",
        filter_post_data_parameters=[
            ("login", DUMMY_USER),
            ("password", DUMMY_PASSWORD),
        ],
        allow_playback_repeats=True,
    ):
        yield SCSession(cluster_config=fxt_server_config)


@pytest.fixture(scope="module")
def fxt_client(fxt_vcr, fxt_server_config: ClusterConfig) -> SCRESTClient:
    """
    This fixture returns an SCRESTClient instance which has already performed
    authentication and retrieved a default workspace id
    """
    with fxt_vcr.use_cassette(
        f"client.{CASSETTE_EXTENSION}",
        filter_post_data_parameters=[
            ("login", DUMMY_USER),
            ("password", DUMMY_PASSWORD),
        ],
    ):
        yield SCRESTClient(
            host=fxt_server_config.host,
            username=fxt_server_config.username,
            password=fxt_server_config.password,
            verify_certificate=fxt_server_config.has_valid_certificate,
            proxies=fxt_server_config.proxies,
        )


@pytest.fixture(scope="module")
def fxt_client_no_vcr(fxt_server_config: ClusterConfig) -> SCRESTClient:
    yield SCRESTClient(
        host=fxt_server_config.host,
        username=fxt_server_config.username,
        password=fxt_server_config.password,
    )
