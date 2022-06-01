from typing import Callable

import pytest

from sc_api_tools import SCRESTClient
from sc_api_tools.rest_managers import ProjectManager
from tests.helpers import ProjectService, force_delete_project


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
    fxt_client: SCRESTClient,
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


@pytest.fixture(scope="class")
def fxt_project_service_no_vcr(fxt_client_no_vcr: SCRESTClient) -> ProjectService:
    project_service = ProjectService(
        client=fxt_client_no_vcr, vcr=None
    )
    yield project_service
    # project_service.delete_project()


@pytest.fixture(scope="class")
def fxt_project_finalizer(fxt_project_manager: ProjectManager) -> Callable[[str], None]:
    """
    This fixture returns a finalizer to ensure project deletion

    :var project_name: Name of the project for which to add the finalizer
    """
    def _project_finalizer(project_name: str) -> None:
        force_delete_project(project_name, fxt_project_manager)
    return _project_finalizer
