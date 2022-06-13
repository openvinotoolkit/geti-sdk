""" This module defines the test configuration """
import os
import shutil
import tempfile

import pytest
from _pytest.main import Session
from sc_api_tools import SCRESTClient

from sc_api_tools.http_session import ClusterConfig
from tests.helpers.project_helpers import remove_all_test_projects

from .helpers import (
    SdkTestMode,
    get_sdk_fixtures,
    replace_host_name_in_cassettes,
)
from .helpers.constants import BASE_TEST_PATH, CASSETTE_PATH, RECORD_CASSETTE_KEY

pytest_plugins = get_sdk_fixtures()

TEST_MODE = SdkTestMode[os.environ.get("TEST_MODE", "OFFLINE")]
HOST = os.environ.get("SC_HOST", "https://dummy_host").strip("/")
USERNAME = os.environ.get("SC_USERNAME", "dummy_user")
PASSWORD = os.environ.get("SC_PASSWORD", "dummy_password")
CLEAR_EXISTING_TEST_PROJECTS = os.environ.get(
    "CLEAR_EXISTING_TEST_PROJECTS", '0'
).lower() in ['true', '1']
NIGHTLY_TEST_LEARNING_PARAMETER_SETTINGS = os.environ.get(
    "LEARNING_PARAMETER_SETTINGS", "default"
)


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
def fxt_base_test_path() -> str:
    """
    This fixture returns the absolute path to the `tests` folder
    """
    yield BASE_TEST_PATH


@pytest.fixture(scope="session")
def fxt_test_mode() -> SdkTestMode:
    """
    This fixture returns the SdkTestMode with which the tests are run
    :return:
    """
    yield TEST_MODE


@pytest.fixture(scope="session")
def fxt_learning_parameter_settings() -> str:
    """
    This fixture returns the settings to use in the nightly test learning parameters.
    Can be either:
      - 'minimal' (single epoch, batch_size = 1),
      - 'reduced_mem' (normal epochs but reduced batch size for memory hungry algos)
      - 'default' (default settings)

    :return:
    """
    yield NIGHTLY_TEST_LEARNING_PARAMETER_SETTINGS


def pytest_sessionstart(session: Session) -> None:
    """
    This function is called before a pytest test run begins.

    If the tests are run in record mode, this hook sets up a temporary directory to
    record the new cassettes to.

    :param session: Pytest session instance that has just been created
    """
    if CLEAR_EXISTING_TEST_PROJECTS and TEST_MODE != SdkTestMode.OFFLINE:
        remove_all_test_projects(
            SCRESTClient(host=HOST, username=USERNAME, password=PASSWORD)
        )
    if TEST_MODE == SdkTestMode.RECORD:
        record_cassette_path = tempfile.mkdtemp()
        print(f"Cassettes will be recorded to `{record_cassette_path}`.")
        os.environ.update({RECORD_CASSETTE_KEY: record_cassette_path})


def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    """
    This function is called after a pytest test run finishes.

    If the tests are run in record mode, this hook handles saving and cleaning the
    recorded cassettes

    :param session: Pytest session that has just finished
    :param exitstatus: Exitstatus with which the tests completed
    """
    if TEST_MODE == SdkTestMode.RECORD:
        record_cassette_path = os.environ.pop(RECORD_CASSETTE_KEY)
        if exitstatus == 0:
            print("Recording successful! Wrapping up....")
            # Copy recorded cassettes to fixtures/cassettes
            print(
                f"Copying newly recorded cassettes from `{record_cassette_path}` to "
                f"`{CASSETTE_PATH}`."
            )
            if os.path.exists(CASSETTE_PATH):
                shutil.rmtree(CASSETTE_PATH)
            shutil.move(record_cassette_path, CASSETTE_PATH)

            # Scrub hostname from cassettes
            replace_host_name_in_cassettes(HOST)
            print(
                f" Hostname {HOST} was scrubbed from all cassette files successfully."
            )
        else:
            # Clean up any cassettes already recorded
            if os.path.exists(record_cassette_path):
                shutil.rmtree(record_cassette_path)
