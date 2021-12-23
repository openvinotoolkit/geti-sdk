import json
import os
from json.decoder import JSONDecodeError
from typing import List

from sc_api_tools.data_models.enums import TaskType


def get_task_types_by_project_type(project_type: str) -> List[TaskType]:
    """
    Returns a list of task_type for each task in the project pipeline, for a
    certain 'project_type'

    :param project_type:
    :return:
    """
    return [TaskType(task) for task in project_type.split('_to_')]
