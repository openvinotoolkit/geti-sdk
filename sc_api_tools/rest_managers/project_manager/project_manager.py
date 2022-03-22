import copy
import json
import os
from typing import Optional, List, Dict, Any, Union, Tuple, Sequence, Mapping

from sc_api_tools.data_models import Project, TaskType, Task
from sc_api_tools.http_session import SCSession
from sc_api_tools.rest_converters import ProjectRESTConverter
from sc_api_tools.utils import remove_null_fields
from sc_api_tools.utils.project_helpers import get_task_types_by_project_type

from .task_templates import (
    BASE_TEMPLATE,
    CROP_TASK,
    DETECTION_TASK,
    SEGMENTATION_TASK,
    CLASSIFICATION_TASK,
    ANOMALY_CLASSIFICATION_TASK,
    ANOMALY_DETECTION_TASK,
    ANOMALY_SEGMENTATION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    ROTATED_DETECTION_TASK
)


TASK_TYPE_MAPPING = {
    TaskType.CROP: CROP_TASK,
    TaskType.DETECTION: DETECTION_TASK,
    TaskType.SEGMENTATION: SEGMENTATION_TASK,
    TaskType.CLASSIFICATION: CLASSIFICATION_TASK,
    TaskType.ANOMALY_CLASSIFICATION: ANOMALY_CLASSIFICATION_TASK,
    TaskType.ANOMALY_DETECTION: ANOMALY_DETECTION_TASK,
    TaskType.ANOMALY_SEGMENTATION: ANOMALY_SEGMENTATION_TASK,
    TaskType.INSTANCE_SEGMENTATION: INSTANCE_SEGMENTATION_TASK,
    TaskType.ROTATED_DETECTION: ROTATED_DETECTION_TASK
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
            project = self.create_project(
                project_name=project_name, project_type=project_type, labels=labels
            )
        return project

    def create_project(
            self,
            project_name: str,
            project_type: str,
            labels: List[Union[List[str], List[Dict[str, Any]]]]
    ) -> Project:
        """
        Creates a new project with name `project_name` on the cluster, containing
        tasks according to the `project_type` specified. Labels for each task are
        specified in the `labels` parameter, which should be a nested list (each entry
         in the outermost list corresponds to the labels for one of the tasks in the
         project pipeline)

        :param project_name: Name of the project
        :param project_type: Type of the project
        :param labels: Nested list of labels
        :raises ValueError: If a project with name `project_name` already exists in
            the workspace
        :return: Project object, as created on the cluster
        """
        project = self.get_project_by_name(project_name)
        if project is not None:
            raise ValueError(
                f"Project with name '{project_name}' already exists, unable to create "
                f"project."
            )
        else:
            project_template = self._create_project_template(
                project_name=project_name, project_type=project_type, labels=labels
            )
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
            json.dump(project_data, file, indent=4)
        print(
            f"Project parameters for project '{project_name}' were saved to file "
            f"{project_config_path}."
        )

    @staticmethod
    def _add_task(
            project_template: dict,
            task_type: TaskType,
            labels: Union[List[str], List[Dict[str, Any]]],
    ) -> Tuple[dict, dict]:
        """
        Adds a task to the pipeline in a project template in dictionary form

        :param project_template: Dictionary representing the project creation data
        :param task_type: Type of the task to be added
        :param labels: Labels to be used for the task
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

        # Make sure the task title is unique in the pipeline
        task_titles_in_template = [task["title"] for task in tasks]
        unique_task_title = ProjectManager._ensure_unique_task_name(
            task_template["title"], task_titles_in_template
        )
        task_template["title"] = unique_task_title

        if not task_type.is_anomaly:
            label_group_name = f"{unique_task_title.lower()} label group"

            for label in labels:
                if isinstance(label, str):
                    label_info = {"name": label, "group": label_group_name}
                else:
                    label_info = label
                task_template["labels"].append(label_info)
        tasks.append(task_template)
        return new_template, task_template

    @staticmethod
    def _add_crop_task(project_template: dict) -> Tuple[dict, dict]:
        """
        Adds a `crop` task to the pipeline in the project_template

        :param project_template:
        :return: Tuple containing:
            - A dictionary representing the new project_template, with the crop task
                added to it
            - A dictionary representing the crop task that was added to the template
        """
        new_template = copy.deepcopy(project_template)
        tasks = new_template["pipeline"]["tasks"]
        crop_task = copy.deepcopy(CROP_TASK)
        tasks.append(crop_task)
        return new_template, crop_task

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

    @staticmethod
    def is_project_dir(path_to_folder: str) -> bool:
        """
        Returns True if the folder specified in `path_to_folder` is a directory
        containing valid SC project data that can be used to upload to an SC cluster

        :param path_to_folder: Directory to check
        :return: True if the directory holds project data, False otherwise
        """
        if not os.path.isdir(path_to_folder):
            return False
        path_to_project = os.path.join(path_to_folder, "project.json")
        if not os.path.isfile(path_to_project):
            return False
        try:
            with open(path_to_project, 'r') as file:
                project_data = json.load(file)
        except json.decoder.JSONDecodeError:
            return False
        try:
            project = ProjectRESTConverter.from_dict(project_data)
        except (ValueError, TypeError, KeyError):
            return False
        return True

    def list_projects(self) -> List[Project]:
        """
        This method prints an overview of all projects that currently exists on the
        cluster, in the workspace managed by the ProjectManager

        NOTE: While this method also returns a list of all the projects, it is
        primarily meant to be used in an interactive environment, such as a
        Jupyter Notebook.

        :return: List of all Projects on the cluster. The returned list is the same as
            the list returned by the `get_all_projects` method
        """
        projects = self.get_all_projects()
        print(f"{len(projects)} projects were found on the platform:\n")
        for project in projects:
            print(" " + project.summary + "\n")
        return projects

    def _create_project_template(
            self,
            project_name: str,
            project_type: str,
            labels: List[Union[List[str], List[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """
        Creates a template dictionary with data for project creation that is ready to
        be sent to  the cluster.

        :param project_name: Name of the project
        :param project_type: Type of the project
        :param labels: Nested list of labels
        :return: Dictionary containing the data to create a project named
            `project_name`, of type `project_type` and with labels `labels`
        """
        project_template = copy.deepcopy(BASE_TEMPLATE)
        previous_task_name = "Dataset"
        previous_task_type = TaskType.DATASET
        task_names_in_template: List[str] = [previous_task_name]
        is_first_task = True
        for task_type, task_labels in zip(
                get_task_types_by_project_type(project_type), labels
        ):
            if not is_first_task and not previous_task_type.is_global:
                # Add crop task and connections, only for tasks that are not
                # first in the pipeline and are not preceded by a global task
                project_template, crop_task = self._add_crop_task(project_template)
                unique_task_name = self._ensure_unique_task_name(
                    crop_task["title"], task_names_in_template
                )
                crop_task["title"] = unique_task_name

                project_template = self._add_connection(
                    project_template,
                    to_task=unique_task_name,
                    from_task=previous_task_name
                )
                previous_task_name = unique_task_name
            project_template, added_task = self._add_task(project_template,
                                                          task_type=task_type,
                                                          labels=task_labels)
            task_name = added_task["title"]

            project_template = self._add_connection(project_template,
                                                    to_task=task_name,
                                                    from_task=previous_task_name)
            previous_task_name = task_name
            previous_task_type = task_type
            is_first_task = False

        project_template["name"] = project_name
        return project_template

    @staticmethod
    def _ensure_unique_task_name(
            task_name: str, task_names_in_template: List[str]
    ) -> str:
        """
        This method checks that the `task_name` passed is not already in the list of
        `task_names_in_template`. If the task_name is already in the list, this method
        will generate a new task_name that is unique

        NOTE: This method updates the list of task_names_in_template with the unique
        task_name that is ensured in this method. This update is done in-place

        :param task_name: Task name to check
        :param task_names_in_template: List of task names that are already taken in the
            project template
        :return: Name of the task that is guaranteed to be unique in the template
        """
        ii = 2
        if task_name in task_names_in_template:
            new_task_name = f"{task_name} {ii}"
            while new_task_name in task_names_in_template:
                ii += 1
                new_task_name = f'{task_name} {ii}'
            task_name = new_task_name
        task_names_in_template.append(task_name)
        return task_name

    def delete_project(
            self, project: Union[str, Project], requires_confirmation: bool = True
    ) -> None:
        """
        Deletes a project. The `project` to delete can either by a Project object or a
        string containing the name of the project to delete.

        By default, this method will ask for user confirmation before deleting the
        project. This can be overridden by passing `requires_confirmation = False`.

        :param project: Project to delete, either a string containing the project
            name or a Project instance
        :param requires_confirmation: True to ask for user confirmation before
            deleting the project, False to delete without confirmation. Defaults to
            True
        """
        if isinstance(project, str):
            project = self.get_project_by_name(project_name=project)
        if not isinstance(project, Project):
            raise TypeError(f"{type(project)} is not a valid project type.")

        if requires_confirmation:
            media_response = self.session.get_rest_response(
                url=f"{self.base_url}projects/{project.id}/datasets/"
                    f"{project.datasets[0].id}/media",
                method="GET"
            )

            media_count = media_response.get("media_count", {"images": 0, "videos": 0})
            user_confirmation = input(
                f"CAUTION: You are about to delete project '{project.name}', "
                f"containing {media_count['images']} images and {media_count['videos']}"
                f" videos, from the platform. Are you sure you want to continue? Type "
                f"Y or YES to continue, any other key to cancel."
            )
            if not (
                    user_confirmation.lower() == "yes"
                    or user_confirmation.lower() == "y"
            ):
                print("Aborting project deletion.")
                return
        try:
            self.session.get_rest_response(
                url=f"{self.base_url}projects/{project.id}",
                method="DELETE"
            )
        except ValueError as error:
            if error.args[-1] == 409:
                raise ValueError(
                    f"Project {project.name} is locked for deletion/modification. "
                    f"Please wait until all jobs related to this project are finished "
                    f"or cancel them to allow deletion/modification."
                )
            else:
                raise error
        print(f"Project '{project.name}' deleted successfully.")

    def add_labels(
        self,
        labels: Union[List[str], List[Dict[str, Any]]],
        project: Project,
        task: Optional[Task] = None
    ) -> Project:
        """
        Adds the `labels` to the project labels. For a project with multiple tasks,
        the `task` parameter can be used to specify the task for which the labels
        should be added.

        :param labels: List of labels to add. Can either be a list of strings
            representing label names, or a list of dictionaries representing label
            properties
        :param project: Project to which the labels should be added
        :param task: Optional Task to add the labels for. Can be left as None for a
            single task project, but is required for a task chain project
        :return: Updated Project instance with the new labels added to it
        """
        # Validate inputs and server version
        if self.session.version < '1.1':
            raise ValueError(
                f"Your server is running SonomaCreek version {self.session.version}. "
                f"Unfortunately this version does not support adding labels, please "
                f"upgrade to SC1.1 or higher."
            )
        if task is not None and task not in project.get_trainable_tasks():
            raise ValueError(
                f"The provided task {task} is not part of project {project}."
            )
        if len(project.get_trainable_tasks()) > 1 and task is None:
            raise ValueError(
                f"Project '{project}' is a task-chain project, but no target task was "
                f"specified. Please provide a valid task to perform label addition."
            )
        task_index = 0 if task is None else project.get_trainable_tasks().index(task)

        # Update the list of labels for the task
        label_list = project.get_labels_per_task()[task_index]
        formatted_labels: List[Dict[str, Any]] = []
        for label_data in labels:
            if isinstance(label_data, str):
                label_dict = {
                    "name": label_data,
                    "color": "#000000",
                    "group": label_data
                }
            elif isinstance(label_data, dict):
                label_name = label_data.get("name", None)
                if label_name is None:
                    raise ValueError(
                        f"Unable to add label {label_data}: Label name not specified."
                    )
                if "color" not in label_data:
                    label_data.update({"color": "#000000"})
                if "group" not in label_data:
                    label_data.update({"group": label_name})
                label_dict = label_data
            else:
                raise ValueError(
                    f"Invalid input label format found for label {label_data}. Please "
                    f"provide either the label name as a string or a dictionary of "
                    f"label properties."
                )
            formatted_labels.append(label_dict)
        label_list.extend(formatted_labels)

        # Prepare data for the update request
        project_data = project.to_dict()
        task_id = project.get_trainable_tasks()[task_index].id
        task_data = next(
            (
                task for task in project_data["pipeline"]["tasks"]
                if task["id"] == task_id
            )
        )
        task_data["labels"] = label_list
        remove_null_fields(project_data)
        print(project_data)
        response = self.session.get_rest_response(
                url=f"{self.base_url}projects/{project.id}",
                method="PUT",
                data=project_data
            )
        return ProjectRESTConverter.from_dict(response)

