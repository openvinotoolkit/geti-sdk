import copy
import json
import os
from typing import Optional, List, Dict, Any, Union

from sc_api_tools.http_session import SCSession

NON_TRAINABLE_TASK_TYPES = ["dataset", "crop"]


class ProjectManager:
    """
    Class to get or create a project in a certain workspace
    """
    SUPPORTED_TYPES = ["detection", "segmentation", "detection_to_segmentation"]

    BASE_TEMPLATE = {
        "name": "dummy project name",
        "pipeline": {
            "connections": [],
            "tasks": [
                {
                    "title": "Dataset",
                    "task_type": "dataset"
                }
            ]
        }

    }

    DETECTION_TASK = {
        "title": "Detection task",
        "task_type": "detection",
        "labels": []
    }

    SEGMENTATION_TASK = {
        "title": "Segmentation task",
        "task_type": "segmentation",
        "labels": []
    }

    CROP_TASK = {
        "title": "Crop task",
        "task_type": "crop"
    }

    def __init__(self, session: SCSession, workspace_id: str):
        self.session = session
        self.base_url = f"workspaces/{workspace_id}/"

    def get_project_by_name(self, project_name: str) -> Optional[dict]:
        project_list = self.session.get_rest_response(
            url=f"{self.base_url}projects/",
            method="GET",
        )
        project = next(
            (project for project in project_list["items"]
             if project["name"] == project_name), None
        )
        if project is None:
            return None
        else:
            return project

    def get_label_names_per_task(
            self, project: Dict[str, Any]
    ) -> List[Dict[str, List[str]]]:
        """
        Retrieves the label names per task in the project

        :param project: Dictionary containing project info as returned by the SC
            clusters' /projects endpoint
        :return: List of dictionaries, mapping the task type for each task to its
            label names. Each dictionary represents one task in the pipeline. The
            list is ordered.
        """
        project_info = self.get_project_parameters(project)
        task_types = self.get_task_types_by_project_type(project_info["project_type"])
        label_names = [
            project_info["label_names_task_one"], project_info["label_names_task_two"]
        ]
        label_names_per_task: List[Dict[str, List[str]]] = []
        for index, task_type in enumerate(task_types):
            label_names_per_task.append({task_type: label_names[index]})
        return label_names_per_task


    @classmethod
    def get_task_types_by_project_type(cls, project_type: str) -> List[str]:
        if project_type not in cls.SUPPORTED_TYPES:
            raise ValueError(
                f"Invalid project type {project_type} specified. Supported types "
                f"are currently {cls.SUPPORTED_TYPES}"
            )
        task_types: List[str] = []
        if project_type == "detection":
            task_types = ["detection"]
        elif project_type == "segmentation":
            task_types = ["segmentation"]
        elif project_type == "detection_to_segmentation":
            task_types = ["detection", "segmentation"]
        return task_types

    def get_or_create_project(
            self,
            project_name: str,
            project_type: str,
            label_names_task_one: List[str],
            label_names_task_two: Optional[List[str]] = None
    ) -> dict:
        if project_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Invalid project type {project_type} specified. Supported types "
                f"are currently {self.SUPPORTED_TYPES}"
            )
        project = self.get_project_by_name(project_name)
        if project is not None:
            print(f"Project with name {project_name} already exists, continuing with "
                  f"exiting project. No new project has been created.")
        else:
            template = copy.deepcopy(self.BASE_TEMPLATE)
            if project_type == "detection":
                project_template = self.add_task(
                    template, task_type="detection", labels=label_names_task_one
                )
                project_template = self.add_connection(
                    project_template, to_task="Detection task", from_task="Dataset"
                )
            elif project_type == "segmentation":
                project_template = self.add_task(
                    template, task_type="segmentation", labels=label_names_task_one
                )
                project_template = self.add_connection(
                    project_template, to_task="Segmentation task", from_task="Dataset"
                )
            elif project_type == "detection_to_segmentation":
                project_template = self.add_task(
                    template, task_type="detection", labels=label_names_task_one
                )
                project_template = self.add_connection(
                    project_template, to_task="Detection task", from_task="Dataset"
                )
                project_template = self.add_crop_task(project_template)
                project_template = self.add_connection(
                    project_template, to_task="Crop task", from_task="Detection task"
                )
                project_template = self.add_task(
                    project_template,
                    task_type="segmentation",
                    labels=label_names_task_two
                )
                project_template = self.add_connection(
                    project_template, to_task="Segmentation task", from_task="Crop task"
                )
            else:
                raise ValueError(
                    f"Project creation is not supported for project type {project_type}"
                )
            project_template["name"] = project_name
            project = self.session.get_rest_response(
                url=f"{self.base_url}projects",
                method="POST",
                data=project_template
            )
            print("Project created successfully.")
        return project

    @staticmethod
    def get_project_parameters(
            project: Dict[str, Any]
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Gets the parameters that can be used to create the project with the
        `ProjectManager.get_or_create_project` method, from a dictionary description
        of that project returned by the SC cluster.

        :param project: Dictionary containing information about a project, returned
            from the project REST endpoint
        :return: Dictionary containing the parameters to re-create the project, using
            the `ProjectManager.get_or_create_project` method
        """
        project_name = project.get("name", None)
        pipeline = project.get("pipeline", None)
        if project_name is None or pipeline is None:
            raise ValueError(
                "Unexpected input format. Expected a dictionary with project info "
                "returned by the SC /projects endpoint."
            )
        tasks = pipeline.get("tasks", None)
        if tasks is None:
            raise ValueError("No tasks found in pipeline, unable to process input.")
        trainable_tasks = [
            task for task in tasks if task["task_type"] not in NON_TRAINABLE_TASK_TYPES
        ]
        if len(trainable_tasks) > 2:
            raise ValueError(
                "Project contains more than 2 trainable tasks, this is not supported "
                "by the ProjectManager at the moment"
            )
        task_types = [task["task_type"] for task in trainable_tasks]
        if len(task_types) == 1:
            project_type = task_types[0]
        else:
            project_type = f"{task_types[0]}_to_{task_types[1]}"

        task_labels = [task["labels"] for task in trainable_tasks]
        label_names_task_one = [
            label["name"] for label in task_labels[0] if not label["is_empty"]
        ]
        label_names_task_two = [
            label["name"] for label in task_labels[1] if not label["is_empty"]
        ]

        return {
            "project_name": project_name,
            "project_type": project_type,
            "label_names_task_one": label_names_task_one,
            "label_names_task_two": label_names_task_two
        }

    def download_project_parameters(
            self, project_name: str, path_to_folder: str
    ) -> None:
        """
        For a project on the SC cluster with name `project_name`, this method gets the
        parameters that can be used to create a project with the
        `ProjectManager.get_or_create_project` method. The parameters are retrieved
        from the cluster and saved in the target folder `path_to_folder`.

        :param project_name: Name of the project to retrieve the parameters for
        :param path_to_folder: Target folder to save the project parameters to.
            Parameters will be saved as a .json file named "project_info.json"
        :raises ValueError: If the project with `project_name` is not found on the
            cluster
        """
        project = self.get_project_by_name(project_name)
        if project is None:
            raise ValueError(
                f"Project with name {project_name} was not found on the cluster."
            )
        parameters = self.get_project_parameters(project=project)
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)
        project_config_path = os.path.join(path_to_folder, "project_info.json")
        with open(project_config_path, 'w') as file:
            json.dump(parameters, file)
        print(
            f"Project parameters for project '{project_name}' were saved to file "
            f"{project_config_path}."
        )

    def add_task(
            self, project_template: dict, task_type: str, labels: List[str]
    ) -> dict:
        new_template = copy.deepcopy(project_template)
        tasks = new_template["pipeline"]["tasks"]
        if task_type == "detection":
            task_template = copy.deepcopy(self.DETECTION_TASK)
            label_group_name = 'default_detection'
        elif task_type == "segmentation":
            task_template = copy.deepcopy(self.SEGMENTATION_TASK)
            label_group_name = 'default_segmentation'
        else:
            raise ValueError(f"Task of type {task_type} is currently not supported.")
        for label in labels:
            task_template["labels"].append({"name": label, "group": label_group_name})
        tasks.append(task_template)
        return new_template

    def add_crop_task(self, project_template: dict) -> dict:
        new_template = copy.deepcopy(project_template)
        tasks = new_template["pipeline"]["tasks"]
        tasks.append(self.CROP_TASK)
        return new_template

    @staticmethod
    def add_connection(project_template: dict, to_task: str, from_task: str) -> dict:
        new_template = copy.deepcopy(project_template)
        connections = new_template["pipeline"]["connections"]
        connections.append({"from": from_task, "to": to_task})
        return new_template

    def create_project_from_folder(self, path_to_folder: str) -> Dict[str, Any]:
        """
        Looks for a `project_info.json` file in the folder at `path_to_folder`, and
        creates a project using the parameters provided in this file

        :param path_to_folder: Folder holding the project data
        :return: dictionary containing the project data as returned by the cluster
        """
        path_to_parameters = os.path.join(path_to_folder, "project_info.json")
        if not os.path.isfile(path_to_parameters):
            raise ValueError(
                f"Unable to find project configuration file at {path_to_parameters}. "
                f"Please provide a valid path to the folder holding the project data."
            )
        with open(path_to_parameters, 'r') as file:
            project_data = json.load(file)
        print(
            f"Creating project '{project_data['project_name']}' from parameters in "
            f"configuration file at {path_to_parameters}."
        )
        return self.get_or_create_project(**project_data)
