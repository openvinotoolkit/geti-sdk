""" This module defines the test configuration """
import os
from typing import Dict, Any

import pytest
from _pytest.main import Session

from vcr.record_mode import RecordMode

from sc_api_tools.http_session import ClusterConfig

from .helpers import (
    TestMode,
    get_sdk_fixtures,
    are_cassettes_available,
    replace_host_name_in_cassettes,
)


BASE_TEST_PATH = os.path.dirname(os.path.abspath(__file__))

pytest_plugins = get_sdk_fixtures()

TEST_MODE = TestMode[os.environ.get("TEST_MODE", "OFFLINE")]
HOST = os.environ.get("SC_HOST", "https://dummy_host").strip("/")
USERNAME = os.environ.get("SC_USERNAME", "dummy_user")
PASSWORD = os.environ.get("SC_PASSWORD", "dummy_password")


@pytest.fixture(scope="session")
def fxt_server_config() -> ClusterConfig:
    """
    This fixture holds the login configuration to access the SC server
    """
    test_config = ClusterConfig(
        host=HOST,
        username=USERNAME,
        password=PASSWORD
    )
    yield test_config


@pytest.fixture(scope="session")
def vcr_record_config() -> Dict[str, Any]:
    """
    This fixture determines the record mode for the VCR cassettes used in the tests
    """
    if TEST_MODE == TestMode.RECORD:
        if are_cassettes_available():
            raise ValueError(
                "Tests were set to run in RECORD mode, but several cassettes were "
                "already found on the system. Please remove all old cassettes (by "
                "deleting 'fixtures/cassettes') before recording a new set."
            )
        yield {"record_mode": RecordMode.NEW_EPISODES}
    elif TEST_MODE == TestMode.ONLINE:
        host = HOST.strip("https://").strip("/")
        yield {"record_mode": RecordMode.NONE, "ignore_hosts": [host]}
    elif TEST_MODE == TestMode.OFFLINE:
        if not are_cassettes_available():
            raise ValueError(
                "Tests were set to run in OFFLINE mode, but no cassettes were found "
                "on the system. Please make sure that the cassettes for the SDK test "
                "suite are available in 'fixtures/cassettes'."
            )
        yield {"record_mode": RecordMode.NONE}
    else:
        raise NotImplementedError(f"TestMode {TEST_MODE} is not implemented")


@pytest.fixture(scope="session")
def base_test_path() -> str:
    """
    This fixture returns the absolute path to the `tests` folder
    """
    yield BASE_TEST_PATH


def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    """
    This function is called after a pytest test run finishes.

    :param session: Pytest session instance
    :param exitstatus: Exitstatus with which the tests completed
    :return:
    """
    if TEST_MODE == TestMode.RECORD:
        replace_host_name_in_cassettes(HOST)
        print(f" Hostname {HOST} was replaced in all cassette files successfully.")
