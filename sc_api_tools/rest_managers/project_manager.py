import copy
from typing import Optional, List, Dict

from sc_api_tools.http_session import SCSession


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
