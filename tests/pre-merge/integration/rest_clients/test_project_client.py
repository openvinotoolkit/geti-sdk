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

from typing import Dict, List

import pytest

from geti_sdk import Geti
from geti_sdk.data_models import Project, TaskType
from geti_sdk.rest_clients import ProjectClient
from tests.helpers.constants import PROJECT_PREFIX
from tests.helpers.project_service import ProjectService


class TestProjectClient:
    @pytest.mark.vcr()
    def test_create_and_delete_project(
        self, fxt_default_labels: List[str], fxt_project_service: ProjectService
    ):
        """
        Verifies that creating and deleting a project through the project client
        works as expected

        Test steps:
        1. Initialize project client
        2. Create classification project
        3. Assert that project name and labels are set correctly
        4. Assert that attempting to create a project with the same name will result
            in a ValueError
        5. Assert that calling the `project_client.get_or_create_project` with the
            same project_name returns the project
        6. Delete project
        7. Assert that attempting to get the project by name after it has been deleted
            will return 'None'
        """
        project = fxt_project_service.create_project()
        assert isinstance(project, Project)

        label_names = [label.name for label in project.get_all_labels()]
        for label_name in fxt_default_labels:
            assert label_name in label_names

        project_client = fxt_project_service.project_client
        with pytest.raises(ValueError):
            project_client.create_project(
                project_name=project.name, project_type="detection", labels=[["none"]]
            )

        project_get_or_create = project_client.get_or_create_project(
            project_name=project.name, project_type="detection", labels=[["none"]]
        )
        assert project_get_or_create.name == project.name
        project_task = project_get_or_create.get_trainable_tasks()[0]
        assert project_task.task_type == TaskType.CLASSIFICATION

        project_client.delete_project(project, requires_confirmation=False)
        assert project_client.get_project_by_name(project.name) is None
        fxt_project_service.reset_state()

    @pytest.mark.vcr()
    def test_get_all_projects(self, fxt_geti: Geti):
        """
        Verifies that getting a list of all projects in the workspace works as expected

        Test steps:
        1. Initialize project client
        2. Retrieve a list of projects
        3. Verify that each entry in the list is a properly deserialized Project
            instance
        """
        project_client = ProjectClient(
            session=fxt_geti.session, workspace_id=fxt_geti.workspace_id
        )
        projects = project_client.get_all_projects()
        for project in projects:
            assert isinstance(project, Project)

    @pytest.mark.vcr()
    def test_hierarchical_classification_project(
        self,
        fxt_project_service: ProjectService,
        fxt_hierarchical_classification_labels: List[Dict[str, str]],
    ):
        """
        Verifies that creating a classification project with hierarchical labels works
        as expected
        """
        project = fxt_project_service.create_project(
            project_name=f"{PROJECT_PREFIX}_project_hierarchical",
            project_type="classification",
            labels=[fxt_hierarchical_classification_labels],
        )
        assert isinstance(project, Project)

        label_names = [label.name for label in project.get_all_labels()]
        for label_rest_data in fxt_hierarchical_classification_labels:
            assert label_rest_data["name"] in label_names
            label = next(
                lab
                for lab in project.get_all_labels()
                if lab.name == label_rest_data["name"]
            )
            if label_rest_data.get("parent_id", None):
                parent_label = next(
                    lab for lab in project.get_all_labels() if lab.id == label.parent_id
                )
                assert parent_label.name == label_rest_data["parent_id"]
            if label_rest_data.get("group", None):
                assert label.group == label_rest_data["group"]