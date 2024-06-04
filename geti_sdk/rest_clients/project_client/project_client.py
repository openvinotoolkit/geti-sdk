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

import copy
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from geti_sdk.data_models import Project, Task, TaskType
from geti_sdk.data_models.utils import remove_null_fields
from geti_sdk.http_session import GetiRequestException, GetiSession
from geti_sdk.rest_clients.dataset_client import DatasetClient
from geti_sdk.rest_converters import ProjectRESTConverter
from geti_sdk.utils.label_helpers import generate_unique_label_color
from geti_sdk.utils.project_helpers import get_task_types_by_project_type

from .task_templates import (
    ANOMALY_CLASSIFICATION_TASK,
    ANOMALY_DETECTION_TASK,
    ANOMALY_SEGMENTATION_TASK,
    BASE_TEMPLATE,
    CLASSIFICATION_TASK,
    CROP_TASK,
    DETECTION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    ROTATED_DETECTION_TASK,
    SEGMENTATION_TASK,
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
    TaskType.ROTATED_DETECTION: ROTATED_DETECTION_TASK,
}


class ProjectClient:
    """
    Class to manipulate projects on the Intel® Geti™ server, within a certain workspace.
    """

    def __init__(self, session: GetiSession, workspace_id: str):
        self.session = session
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/"

    def get_all_projects(
        self, request_page_size: int = 50, get_project_details: bool = True
    ) -> List[Project]:
        """
        Return a list of projects found on the Intel® Geti™ server

        :param request_page_size: Max number of projects to fetch in a single HTTP
            request. Higher values may reduce the response time of this method when
            there are many projects, but increase the chance of timeout.
        :param get_project_details: True to get all details of the projects on the
            Intel® Geti™, False to fetch only a summary of each project. Set this to
            False if minimizing latency is a concern. Defaults to True
        :return: List of Project objects, containing the project information for each
            project on the Intel® Geti™ server
        """
        project_key = "projects"
        num_total_projects_key = "project_counts"

        # The 'projects' endpoint uses pagination: multiple HTTP may be necessary to
        # fetch the full list of projects
        project_rest_list: List[Dict] = []
        while response := self.session.get_rest_response(
            url=f"{self.base_url}projects?limit={request_page_size}&skip={len(project_rest_list)}",
            method="GET",
        ):
            project_rest_list.extend(response[project_key])
            if len(project_rest_list) >= response[num_total_projects_key]:
                break

        project_list = [
            ProjectRESTConverter.from_dict(project_input=project)
            for project in project_rest_list
        ]
        if get_project_details:
            project_detail_list: List[Project] = []
            for project in project_list:
                try:
                    project_detail_list.append(self._get_project_detail(project))
                except GetiRequestException as e:
                    if e.status_code == 403:
                        logging.info(
                            f"Unable to access project `{project.name}` details: Unauthorized."
                        )
                        project_detail_list.append(project)
            return project_detail_list
        else:
            return project_list

    def get_project_by_name(
        self, project_name: str, project_id: Optional[str] = None
    ) -> Optional[Project]:
        """
        Get a project from the Intel® Geti™ server by project_name.

        :param project_name: Name of the project to get
        :param project_id: Optional ID of the project to get. Only used if more than
            one project named `project_name` exists in the workspace.
        :raises: ValueError in case multiple projects with the specified name exist on
            the server, and no `project_id` is provided in order to allow unique
            identification of the project.
        :return: Project object containing the data of the project, if the project is
            found on the server. Returns None if the project doesn't exist
        """
        project_list = self.get_all_projects(get_project_details=False)
        matches = [project for project in project_list if project.name == project_name]
        if len(matches) == 1:
            return self._get_project_detail(matches[0])
        elif len(matches) > 1:
            if project_id is None:
                detailed_matches = [
                    self._get_project_detail(match) for match in matches
                ]
                projects_info = [
                    (
                        f"Name: {p.name}, Type: {p.project_type}, ID: {p.id}, "
                        f"creation_date: {p.creation_time}"
                    )
                    for p in detailed_matches
                ]
                raise ValueError(
                    f"A total of {len(matches)} projects named `{project_name}` were "
                    f"found in the workspace. Unable to uniquely identify the "
                    f"desired project. Please provide a `project_id` to ensure the "
                    f"proper project is returned. The following projects were found:"
                    f"{projects_info}"
                )
            else:
                matched_project = next(
                    (project for project in matches if project.id == project_id), None
                )
                if matched_project is None:
                    logging.info(
                        f"Projects with name `{project_name}` were found, but none of "
                        f"the project ID's `{[p.id for p in matches]}` matches the "
                        f"requested id `{project_id}`."
                    )
                return matched_project
        else:
            return None

    def get_or_create_project(
        self,
        project_name: str,
        project_type: str,
        labels: List[Union[List[str], List[Dict[str, Any]]]],
    ) -> Project:
        """
        Create a new project with name `project_name` on the cluster, or retrieve
        the data for an existing project with `project_name` if it exists.

        :param project_name: Name of the project
        :param project_type: Type of the project
        :param labels: Nested list of labels
        :return:
        """
        project = self.get_project_by_name(project_name)
        if project is not None:
            logging.info(
                f"Project with name {project_name} exists, continuing with "
                f"exiting project. No new project has been created."
            )
        else:
            project = self.create_project(
                project_name=project_name, project_type=project_type, labels=labels
            )
        return project

    def create_project(
        self,
        project_name: str,
        project_type: str,
        labels: List[Union[List[str], List[Dict[str, Any]]]],
    ) -> Project:
        """
        Create a new project with name `project_name` on the cluster, containing
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
        project_template = self._create_project_template(
            project_name=project_name, project_type=project_type, labels=labels
        )
        project = self.session.get_rest_response(
            url=f"{self.base_url}projects", method="POST", data=project_template
        )
        logging.info("Project created successfully.")
        project = ProjectRESTConverter.from_dict(project)
        self._await_project_ready(project=project)
        return project

    def download_project_info(self, project_name: str, path_to_folder: str) -> None:
        """
        Get the project data that can be used for project creation for a project on
        the Intel® Geti™ server, named `project_name`. From the returned data, the
        method `ProjectClient.get_or_create_project` can create a project on the
        Intel® Geti™ server. The data is retrieved from the cluster and saved in the
        target folder `path_to_folder`.

        :param project_name: Name of the project to retrieve the data for
        :param path_to_folder: Target folder to save the project data to.
            Data will be saved as a .json file named "project.json"
        :raises ValueError: If the project with `project_name` is not found on the
            cluster
        """
        project = self.get_project_by_name(project_name)
        if project is None:
            raise ValueError(
                f"Project with name {project_name} was not found on the server."
            )
        project_data = ProjectRESTConverter.to_dict(project)
        os.makedirs(path_to_folder, exist_ok=True, mode=0o770)
        project_config_path = os.path.join(path_to_folder, "project.json")
        with open(project_config_path, "w") as file:
            json.dump(project_data, file, indent=4)
        logging.info(
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
        Add a task to the pipeline in a project template in dictionary form.

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
        unique_task_title = ProjectClient._ensure_unique_task_name(
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
        Add a `crop` task to the pipeline in the project_template.

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
        Add a connection between `from_task` and `to_task` in the project_template
        dictionary.

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
        Look for a `project.json` file in the folder at `path_to_folder`, and
        create a project using the parameters provided in this file.

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
        with open(path_to_project, "r") as file:
            project_data = json.load(file)
        project = ProjectRESTConverter.from_dict(project_data)
        if project_name is not None:
            project.name = project_name
        logging.info(
            f"Creating project '{project.name}' from parameters in "
            f"configuration file at {path_to_project}."
        )
        project.prepare_for_post()
        datasets = project.datasets
        created_project = self.get_or_create_project(**project.get_parameters())
        if len(datasets) > 1:
            # Create the additional datasets if needed
            dataset_client = DatasetClient(
                session=self.session,
                workspace_id=self.workspace_id,
                project=created_project,
            )
            for dataset in datasets:
                if dataset.name not in [ds.name for ds in created_project.datasets]:
                    dataset_client.create_dataset(name=dataset.name)
        return created_project

    @staticmethod
    def is_project_dir(path_to_folder: str) -> bool:
        """
        Check if the folder specified in `path_to_folder` is a directory
        containing valid Intel® Geti™ project data that can be used to upload to an
        Intel® Geti™ server.

        :param path_to_folder: Directory to check
        :return: True if the directory holds project data, False otherwise
        """
        if not os.path.isdir(path_to_folder):
            return False
        path_to_project = os.path.join(path_to_folder, "project.json")
        if not os.path.isfile(path_to_project):
            return False
        try:
            with open(path_to_project, "r") as file:
                project_data = json.load(file)
        except json.decoder.JSONDecodeError:
            return False
        try:
            ProjectRESTConverter.from_dict(project_data)
        except (ValueError, TypeError, KeyError):
            return False
        return True

    def list_projects(self) -> List[Project]:
        """
        Print an overview of all projects that currently exists on the
        cluster, in the workspace managed by the ProjectClient

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
        labels: List[Union[List[str], List[Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        """
        Create a template dictionary with data for project creation that is ready to
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
                    from_task=previous_task_name,
                )
                previous_task_name = unique_task_name
            project_template, added_task = self._add_task(
                project_template, task_type=task_type, labels=task_labels
            )
            task_name = added_task["title"]

            project_template = self._add_connection(
                project_template, to_task=task_name, from_task=previous_task_name
            )
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
        Check that the `task_name` passed is not already in the list of
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
                new_task_name = f"{task_name} {ii}"
            task_name = new_task_name
        task_names_in_template.append(task_name)
        return task_name

    def delete_project(
        self, project: Union[str, Project], requires_confirmation: bool = True
    ) -> None:
        """
        Delete a project. The `project` to delete can either by a Project object or a
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
                method="GET",
            )

            media_count = media_response.get("media_count", {"images": 0, "videos": 0})
            user_confirmation = input(
                f"CAUTION: You are about to delete project '{project.name}', "
                f"containing {media_count['images']} images and {media_count['videos']}"
                f" videos, from the platform. Are you sure you want to continue? Type "
                f"Y or YES to continue, any other key to cancel."
            )
            if not (
                user_confirmation.lower() == "yes" or user_confirmation.lower() == "y"
            ):
                logging.info("Aborting project deletion.")
                return
        try:
            self.session.get_rest_response(
                url=f"{self.base_url}projects/{project.id}", method="DELETE"
            )
        except GetiRequestException as error:
            if error.status_code == 409:
                raise ValueError(
                    f"Project {project.name} is locked for deletion/modification. "
                    f"Please wait until all jobs related to this project are finished "
                    f"or cancel them to allow deletion/modification."
                )
            else:
                raise error
        logging.info(f"Project '{project.name}' deleted successfully.")

    def add_labels(
        self,
        labels: Union[List[str], List[Dict[str, Any]]],
        project: Project,
        task: Optional[Task] = None,
        revisit_affected_annotations: bool = False,
    ) -> Project:
        """
        Add the `labels` to the project labels. For a project with multiple tasks,
        the `task` parameter can be used to specify the task for which the labels
        should be added.

        :param labels: List of labels to add. Can either be a list of strings
            representing label names, or a list of dictionaries representing label
            properties
        :param project: Project to which the labels should be added
        :param task: Optional Task to add the labels for. Can be left as None for a
            single task project, but is required for a task chain project
        :param revisit_affected_annotations: True to make sure that the server will
            assign a `to_revisit` status to all annotations linked to the label(s)
            that are added. False to not revisit any potentially linked annotations.
        :return: Updated Project instance with the new labels added to it
        """
        # Validate inputs
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
        existing_colors = [label["color"] for label in label_list]
        formatted_labels: List[Dict[str, Any]] = []
        for label_data in labels:
            new_color = generate_unique_label_color(existing_colors)
            if isinstance(label_data, str):
                label_dict = {
                    "name": label_data,
                    "color": new_color,
                    "group": label_data,
                }
            elif isinstance(label_data, dict):
                label_name = label_data.get("name", None)
                if label_name is None:
                    raise ValueError(
                        f"Unable to add label {label_data}: Label name not specified."
                    )
                if "color" not in label_data:
                    label_data.update({"color": new_color})
                if "group" not in label_data:
                    label_data.update({"group": label_name})
                label_dict = label_data
            else:
                raise ValueError(
                    f"Invalid input label format found for label {label_data}. Please "
                    f"provide either the label name as a string or a dictionary of "
                    f"label properties."
                )
            label_dict.update(
                {"revisit_affected_annotations": revisit_affected_annotations}
            )
            formatted_labels.append(label_dict)
            existing_colors.append(new_color)
        label_list.extend(formatted_labels)

        # Prepare data for the update request
        project.prepare_for_post()
        project_data = project.to_dict()
        task_id = project.get_trainable_tasks()[task_index].id
        task_data = next(
            (
                task
                for task in project_data["pipeline"]["tasks"]
                if task["id"] == task_id
            )
        )
        task_data["labels"] = label_list
        remove_null_fields(project_data)
        logging.info(project_data)
        response = self.session.get_rest_response(
            url=f"{self.base_url}projects/{project.id}", method="PUT", data=project_data
        )
        return ProjectRESTConverter.from_dict(response)

    def _await_project_ready(
        self, project: Project, timeout: int = 5, interval: int = 1
    ) -> None:
        """
        Await the completion of the project creation process on the Intel® Geti™ server

        :param project: Project object representing the project
        :param timeout: Time (in seconds) after which the method will time out and
            raise an error
        :param interval: Interval (in seconds) between status checks of the project
        :raises: TimeoutError if the project does not become ready after the specified
            timeout
        """
        t_start = time.time()
        error: Optional[BaseException] = None
        while time.time() - t_start < timeout:
            try:
                self.session.get_rest_response(
                    url=f"{self.base_url}projects/{project.id}/status", method="GET"
                )
                return
            except GetiRequestException as latest_error:
                time.sleep(interval)
                error = latest_error
        raise TimeoutError(
            f"Project has not become ready within the specified timeout ({timeout} "
            f"seconds)."
        ) from error

    def _get_project_detail(self, project: Union[Project, str]) -> Project:
        """
        Fetch the most recent project details from the Intel® Geti™ server

        :param project: Name of the project or Project object representing the project
            to get detailed information for.
        :return: Updated Project object
        """
        if isinstance(project, str):
            project = self.get_project_by_name(project_name=project)
            return project
        else:
            response = self.session.get_rest_response(
                url=f"{self.base_url}projects/{project.id}", method="GET"
            )
            return ProjectRESTConverter.from_dict(response)
