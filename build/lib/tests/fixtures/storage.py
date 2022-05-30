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
