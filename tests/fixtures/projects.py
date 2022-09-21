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

from typing import Callable

import pytest

from geti_sdk import Geti
from geti_sdk.rest_clients import ProjectClient
from tests.helpers import ProjectService, force_delete_project


@pytest.fixture(scope="class")
def fxt_project_client(fxt_geti: Geti) -> ProjectClient:
    """
    This fixture returns a ProjectClient instance corresponding to the Geti instance
    """
    yield ProjectClient(session=fxt_geti.session, workspace_id=fxt_geti.workspace_id)


@pytest.fixture(scope="class")
def fxt_project_service(
    fxt_vcr,
    fxt_geti: Geti,
) -> ProjectService:
    """
    This fixture provides a service for creating a project and the corresponding
    clients to interact with it.

    A project can be created by using `fxt_project_service.create_project()`, which
    takes various parameters

    The project is deleted once the test function finishes.
    """
    project_service = ProjectService(geti=fxt_geti, vcr=fxt_vcr)
    yield project_service
    project_service.delete_project()


@pytest.fixture(scope="class")
def fxt_project_service_no_vcr(fxt_geti_no_vcr: Geti) -> ProjectService:
    project_service = ProjectService(geti=fxt_geti_no_vcr, vcr=None)
    yield project_service
    # project_service.delete_project()


@pytest.fixture(scope="class")
def fxt_project_finalizer(fxt_project_client: ProjectClient) -> Callable[[str], None]:
    """
    This fixture returns a finalizer to ensure project deletion

    :var project_name: Name of the project for which to add the finalizer
    """

    def _project_finalizer(project_name: str) -> None:
        force_delete_project(project_name, fxt_project_client)

    return _project_finalizer
