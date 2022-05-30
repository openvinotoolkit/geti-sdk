import os
from typing import List

from .constants import BASE_TEST_PATH


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
