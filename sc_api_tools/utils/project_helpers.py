import json
import os
from json.decoder import JSONDecodeError
from typing import List

from sc_api_tools.data_models.enums import TaskType
from sc_api_tools.rest_converters import ProjectRESTConverter


def get_task_types_by_project_type(project_type: str) -> List[TaskType]:
    """
    Returns a list of task_type for each task in the project pipeline, for a
    certain 'project_type'

    :param project_type:
    :return:
    """
    return [TaskType(task) for task in project_type.split('_to_')]


def is_project_dir(path_to_folder: str) -> bool:
    """
    Returns True if the folder specified in `path_to_folder` is a directory containing
    valid SC project data that can be used to upload to an SC cluster

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
    except JSONDecodeError:
        return False
    try:
        project = ProjectRESTConverter.from_dict(project_data)
    except (ValueError, TypeError, KeyError):
        return False
    return True
