import os

import pytest
import tempfile
import shutil


@pytest.fixture(scope="class")
def fxt_temp_directory() -> str:
    """
    This fixture returns the path to a temporary directory that can be used for
    storing data for the duration of the tests. The directory is removed after all
    tests in the test class have finished.

    :return: Path to the temporary storage directory
    """
    tempdir = tempfile.mkdtemp()
    yield tempdir
    shutil.rmtree(tempdir)


@pytest.fixture(scope="session")
def fxt_artifact_directory() -> str:
    """
    Returns the path to the directory in which test artifacts should be saved, when the
    tests are run in the CI pipeline.

    :return: Path to the directory for test artifacts
    """
    yield os.environ.get("ARTIFACT_DIRECTORY")
