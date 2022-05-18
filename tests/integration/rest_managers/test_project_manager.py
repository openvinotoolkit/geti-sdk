from typing import Dict, List

import pytest

from sc_api_tools import SCRESTClient
from sc_api_tools.rest_managers import ProjectManager
from sc_api_tools.data_models import Project, TaskType

from tests.helpers.project_service import ProjectService
from tests.helpers.constants import PROJECT_PREFIX


class TestProjectManager:
    @pytest.mark.vcr()
    def test_create_and_delete_project(
            self,
            fxt_default_labels: List[str],
            fxt_project_service: ProjectService
    ):
        """
        Verifies that creating and deleting a project through the project manager
        works as expected

        Test steps:
        1. Initialize project manager
        2. Create classification project
        3. Assert that project name and labels are set correctly
        4. Assert that attempting to create a project with the same name will result
            in a ValueError
        5. Assert that calling the `project_manager.get_or_create_project` with the
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

        project_manager = fxt_project_service.project_manager
        with pytest.raises(ValueError):
            project_manager.create_project(
                project_name=project.name, project_type='detection', labels=[['none']]
            )

        project_get_or_create = project_manager.get_or_create_project(
            project_name=project.name,
            project_type='detection',
            labels=[['none']]
        )
        assert project_get_or_create.name == project.name
        project_task = project_get_or_create.get_trainable_tasks()[0]
        assert project_task.task_type == TaskType.CLASSIFICATION

        project_manager.delete_project(project, requires_confirmation=False)
        assert project_manager.get_project_by_name(project.name) is None
        fxt_project_service.reset_state()

    @pytest.mark.vcr()
    def test_get_all_projects(self, fxt_client: SCRESTClient):
        """
        Verifies that getting a list of all projects in the workspace works as expected

        Test steps:
        1. Initialize project manager
        2. Retrieve a list of projects
        3. Verify that each entry in the list is a properly deserialized Project
            instance
        """
        project_manager = ProjectManager(
            session=fxt_client.session, workspace_id=fxt_client.workspace_id
        )
        projects = project_manager.get_all_projects()
        for project in projects:
            assert isinstance(project, Project)

    @pytest.mark.vcr()
    def test_hierarchical_classification_project(
            self,
            fxt_project_service: ProjectService,
            fxt_hierarchical_classification_labels: List[Dict[str, str]]
    ):
        """
        Verifies that creating a classification project with hierarchical labels works
        as expected
        """
        project = fxt_project_service.create_project(
            project_name=f"{PROJECT_PREFIX}_project_hierarchical",
            project_type="classification",
            labels=[fxt_hierarchical_classification_labels])
        assert isinstance(project, Project)

        label_names = [label.name for label in project.get_all_labels()]
        for label_rest_data in fxt_hierarchical_classification_labels:
            assert label_rest_data["name"] in label_names
            label = next(
                lab for lab in project.get_all_labels()
                if lab.name == label_rest_data["name"]
            )
            if label_rest_data.get("parent_id", None):
                parent_label = next(
                    lab for lab in project.get_all_labels()
                    if lab.id == label.parent_id
                )
                assert parent_label.name == label_rest_data["parent_id"]
            if label_rest_data.get("group", None):
                assert label.group == label_rest_data["group"]
