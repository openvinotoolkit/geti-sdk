import copy
import json
import os
from typing import Optional, List, Dict, Any, Union, Tuple

from sc_api_tools.data_models import Project, TaskType
from sc_api_tools.http_session import SCSession
from sc_api_tools.rest_converters import ProjectRESTConverter
from sc_api_tools.data_models.enums.task_type import ANOMALY_TASK_TYPES

from .task_templates import (
    BASE_TEMPLATE,
    CROP_TASK,
    DETECTION_TASK,
    SEGMENTATION_TASK,
    CLASSIFICATION_TASK,
    ANOMALY_CLASSIFICATION_TASK
)
from ...utils.project_helpers import get_task_types_by_project_type

TASK_TYPE_MAPPING = {
    TaskType.CROP: CROP_TASK,
    TaskType.DETECTION: DETECTION_TASK,
    TaskType.SEGMENTATION: SEGMENTATION_TASK,
    TaskType.CLASSIFICATION: CLASSIFICATION_TASK,
    TaskType.ANOMALY_CLASSIFICATION: ANOMALY_CLASSIFICATION_TASK
}


class ProjectManager:
    """
    Class to get or create a project in a certain workspace
    """
    def __init__(self, session: SCSession, workspace_id: str):
        self.session = session
        self.base_url = f"workspaces/{workspace_id}/"

    def get_all_projects(self) -> List[Project]:
        """
        Returns a list of projects found on the SC cluster

        :return: List of Project objects, containing the project information for each
            project on the SC cluster
        """
        project_list = self.session.get_rest_response(
            url=f"{self.base_url}projects/",
            method="GET",
        )
        return [
            ProjectRESTConverter.from_dict(project_input=project)
            for project in project_list["items"]
        ]

    def get_project_by_name(self, project_name: str) -> Optional[Project]:
        """
        Get a project from the SC cluster by project_name.

        :param project_name: Name of the project to get
        :return: Project object containing the data of the project, if the project is
            found on the cluster. Returns None if the project doesn't exist
        """
        project_list = self.get_all_projects()
        project = next(
            (project for project in project_list
             if project.name == project_name), None
        )
        return project

    @classmethod
    def get_task_types_by_project_type(cls, project_type: str) -> List[TaskType]:
        """
        Returns a list of task_type for each task in the project pipeline, for a
        certain 'project_type'

        :param project_type:
        :return:
        """
        return [TaskType(task) for task in project_type.split('_to_')]

    def get_or_create_project(
            self,
            project_name: str,
            project_type: str,
            labels: List[Union[List[str], List[Dict[str, Any]]]]
    ) -> Project:
        """
        Creates a new project with name `project_name` on the cluster, or retrieves
        the data for an existing project with `project_name` if it exists.

        :param project_name: Name of the project
        :param project_type: Type of the project
        :param labels: Nested list of labels
        :return:
        """
        project = self.get_project_by_name(project_name)
        if project is not None:
            print(f"Project with name {project_name} already exists, continuing with "
                  f"exiting project. No new project has been created.")
        else:
            project_template = copy.deepcopy(BASE_TEMPLATE)
            previous_task_name = "Dataset"
            is_first_task = True
            for task_type, task_labels in zip(
                    get_task_types_by_project_type(project_type), labels
            ):
                if not is_first_task:
                    # Add crop task and connections, only for tasks that are not
                    # first in the pipeline
                    project_template = self._add_crop_task(project_template)
                    task_name = "Crop task"
                    project_template = self._add_connection(
                        project_template,
                        to_task=task_name,
                        from_task=previous_task_name
                    )
                    previous_task_name = task_name
                project_template, added_task = self._add_task(project_template,
                                                              task_type=task_type,
                                                              labels=task_labels)
                task_name = added_task["title"]
                project_template = self._add_connection(project_template,
                                                        to_task=task_name,
                                                        from_task=previous_task_name)
                previous_task_name = task_name
                is_first_task = False

            project_template["name"] = project_name
            project = self.session.get_rest_response(
                url=f"{self.base_url}projects",
                method="POST",
                data=project_template
            )
            print("Project created successfully.")
            project = ProjectRESTConverter.from_dict(project)
        return project

    def download_project_info(
            self, project_name: str, path_to_folder: str
    ) -> None:
        """
        For a project on the SC cluster with name `project_name`, this method gets the
        project data that can be used to create a project with the
        `ProjectManager.get_or_create_project` method. The data is retrieved
        from the cluster and saved in the target folder `path_to_folder`.

        :param project_name: Name of the project to retrieve the data for
        :param path_to_folder: Target folder to save the project data to.
            Data will be saved as a .json file named "project.json"
        :raises ValueError: If the project with `project_name` is not found on the
            cluster
        """
        project = self.get_project_by_name(project_name)
        if project is None:
            raise ValueError(
                f"Project with name {project_name} was not found on the cluster."
            )
        project_data = ProjectRESTConverter.to_dict(project)
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)
        project_config_path = os.path.join(path_to_folder, "project.json")
        with open(project_config_path, 'w') as file:
            json.dump(project_data, file)
        print(
            f"Project parameters for project '{project_name}' were saved to file "
            f"{project_config_path}."
        )

    @staticmethod
    def _add_task(
            project_template: dict,
            task_type: TaskType,
            labels: Union[List[str], List[Dict[str, Any]]]
    ) -> Tuple[dict, dict]:
        """
        Adds a task to the pipeline in a project template in dictionary form

        :param project_template:
        :param task_type:
        :param labels:
        :return:
        """
        new_template = copy.deepcopy(project_template)
        tasks = new_template["pipeline"]["tasks"]
        try:
            task_template = copy.deepcopy(TASK_TYPE_MAPPING[task_type])
        except KeyError as error:
            raise ValueError(
                f"Task of type {task_type} is currently not supported."
            ) from error
        if task_type.value not in ANOMALY_TASK_TYPES:
            label_group_name = f"default_{task_type}"
            for label in labels:
                if isinstance(label, str):
                    label_info = {"name": label, "group": label_group_name}
                else:
                    label_info = label
                task_template["labels"].append(label_info)
        tasks.append(task_template)
        return new_template, task_template

    @staticmethod
    def _add_crop_task(project_template: dict) -> dict:
        """
        Adds a `crop` task to the pipeline in the project_template

        :param project_template:
        :return:
        """
        new_template = copy.deepcopy(project_template)
        tasks = new_template["pipeline"]["tasks"]
        tasks.append(CROP_TASK)
        return new_template

    @staticmethod
    def _add_connection(project_template: dict, to_task: str, from_task: str) -> dict:
        """
        Adds a connection between `from_task` and `to_task` in the project_template
        dictionary

        :param project_template:
        :param to_task:
        :param from_task:
        :return:
        """
        new_template = copy.deepcopy(project_template)
        connections = new_template["pipeline"]["connections"]
        connections.append({"from": from_task, "to": to_task})
        return new_template

    def create_project_from_folder(
            self, path_to_folder: str, project_name: Optional[str] = None
    ) -> Project:
        """
        Looks for a `project.json` file in the folder at `path_to_folder`, and
        creates a project using the parameters provided in this file

        :param path_to_folder: Folder holding the project data
        :param project_name: Optional name of the project. If not specified, the
            project name found in the project configuration in the upload folder
            will be used.
        :return: Project as created on the cluster
        """
        path_to_project = os.path.join(path_to_folder, "project.json")
        if not os.path.isfile(path_to_project):
            raise ValueError(
                f"Unable to find project configuration file at {path_to_project}. "
                f"Please provide a valid path to the folder holding the project data."
            )
        with open(path_to_project, 'r') as file:
            project_data = json.load(file)
        project = ProjectRESTConverter.from_dict(project_data)
        if project_name is not None:
            project.name = project_name
        print(
            f"Creating project '{project.name}' from parameters in "
            f"configuration file at {path_to_project}."
        )
        return self.get_or_create_project(**project.get_parameters())
