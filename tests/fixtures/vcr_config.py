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

""" Configuration fixtures for VCR """
import os
from typing import Any, Dict

import pytest
from vcr import VCR
from vcr.record_mode import RecordMode

from tests.helpers import SdkTestMode, are_cassettes_available
from tests.helpers.constants import (
    CASSETTE_EXTENSION,
    CASSETTE_PATH,
    RECORD_CASSETTE_KEY,
)


@pytest.fixture(scope="session")
def vcr_record_config(fxt_test_mode, fxt_server_config) -> Dict[str, Any]:
    """
    This fixture determines the record mode for the VCR cassettes used in the tests
    """
    if fxt_test_mode == SdkTestMode.RECORD:
        yield {"record_mode": RecordMode.NEW_EPISODES}
    elif fxt_test_mode == SdkTestMode.ONLINE:
        host = fxt_server_config.host.replace("https://", "").strip("/")
        yield {"record_mode": RecordMode.NONE, "ignore_hosts": [host]}
    elif fxt_test_mode == SdkTestMode.OFFLINE:
        if not are_cassettes_available():
            raise ValueError(
                f"Tests were set to run in OFFLINE mode, but no cassettes were found "
                f"on the system. Please make sure that the cassettes for the SDK test "
                f"suite are available in '{CASSETTE_PATH}'."
            )
        yield {"record_mode": RecordMode.NONE}
    else:
        raise NotImplementedError(f"SdkTestMode {fxt_test_mode} is not implemented")


@pytest.fixture(scope="session")
def vcr_config(vcr_record_config) -> Dict[str, Any]:
    """
    This fixture defines the configuration for VCR.py
    """
    vcr_config_dict = {
        "filter_headers": ["authorization"],
        "ignore_localhost": True,
        "path_transformer": VCR.ensure_suffix(f".{CASSETTE_EXTENSION}"),
    }
    vcr_config_dict.update(vcr_record_config)
    return vcr_config_dict


@pytest.fixture(scope="session")
def vcr_cassette_dir(fxt_test_mode) -> str:
    """
    Returns the path to the directory from which cassettes should be read (in offline
    mode), or to which they should be recorded (in record mode).
    """
    if fxt_test_mode == SdkTestMode.RECORD:
        yield os.environ.get(RECORD_CASSETTE_KEY)
    else:
        yield CASSETTE_PATH


@pytest.fixture(scope="session")
def fxt_vcr(vcr_config, vcr_cassette_dir) -> VCR:
    """
    This fixture instantiates a VCR.py instance, according to the configuration
    defined in the `vcr_config` and `vcr_cassette_dir` fixtures
    """
    yield VCR(**vcr_config, cassette_library_dir=vcr_cassette_dir)
