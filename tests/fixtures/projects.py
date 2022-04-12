import pytest

from sc_api_tools import SCRESTClient
from sc_api_tools.rest_managers import ProjectManager
from tests.helpers.project_service import ProjectService


@pytest.fixture(scope="class")
def fxt_project_manager(fxt_client: SCRESTClient) -> ProjectManager:
    """
    This fixture returns a ProjectManager instance corresponding to the client
    """
    yield ProjectManager(
        session=fxt_client.session, workspace_id=fxt_client.workspace_id
    )


@pytest.fixture(scope="class")
def fxt_project_service(
    fxt_vcr,
    fxt_client
) -> ProjectService:
    """
    This fixture provides a service for creating a project and the corresponding
    managers to interact with it.

    A project can be created by using `fxt_project_service.create_project()`, which
    takes various parameters

    The project is deleted once the test function finishes.
    """
    project_service = ProjectService(client=fxt_client, vcr=fxt_vcr)
    yield project_service
    project_service.delete_project()
