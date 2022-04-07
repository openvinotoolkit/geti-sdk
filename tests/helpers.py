import os
from enum import Enum
from typing import List

BASE_TEST_PATH = os.path.dirname(os.path.abspath(__file__))


class TestMode(Enum):
    """
    This Enum represents the different modes available for running the tests. The
    available modes are:
        - ONLINE:  The tests are run against an actual SC server instance. Real http
                     requests are being made but no cassettes are recorded
                     Use this mode to verify that the SDK data models are still up to
                     date with the current SC REST contracts
        - OFFLINE: The tests are run using the recorded requests and responses. All
                     http requests are intercepted an no actual requests will be made.
                     Use this mode in a CI environment, or for fast testing of the SDK
                     logic
        - RECORD:  The tests are run against an actual SC server instance. HTTP requests
                     are being made and all requests and responses are recorded to a new
                     set of cassettes for the SDK test suite. The old cassettes for the
                     tests are deleted.
    """
    ONLINE = 'online'
    OFFLINE = 'offline'
    RECORD = 'record'


def get_sdk_fixtures() -> List[str]:
    """
    Returns the list of fixtures available to the SDK

    :return: list of fixture paths for pytest to import
    """
    fixture_filenames = os.listdir(os.path.join(BASE_TEST_PATH, 'fixtures'))
    fixtures: List[str] = []
    for filename in fixture_filenames:
        if filename.endswith('.py') and not filename.startswith('__'):
            fixtures.append(f"tests.fixtures.{filename[0:-3]}")
    return fixtures


def are_cassettes_available() -> bool:
    """
    Checks that the VCR cassettes required to run the tests offline are available

    :return: True if the cassettes are available in the proper path, False otherwise
    """
    cassette_path = os.path.join(BASE_TEST_PATH, 'fixtures', 'cassettes')
    if not os.path.isdir(cassette_path):
        return False
    if len(os.listdir(cassette_path)) > 0:
        return True
    return False
