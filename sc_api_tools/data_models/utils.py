from enum import Enum

import attr
from typing import Any

from sc_api_tools.data_models import TaskType


def deidentify(instance: Any):
    """
    Sets all identifier fields of an instance of an attr.s decorated class within the
    SC REST DataModels to None

    :param instance: Object to deidentify
    """
    for field in attr.fields(type(instance)):
        if field.name in instance._identifier_fields:
            setattr(instance, field.name, None)


def str_to_task_type(task_type: str) -> TaskType:
    """
    Converts an input string to a task type

    :param task_type:
    :return: TaskType instance corresponding to `task_type`
    """
    return TaskType(task_type)


def attr_enum_to_str(instance, field, value):
    """
    Converts an enum in an attr.s decorated class to string representation, used while
    converting the attrs object to a dictionary.

    :param instance:
    :param field:
    :param value:
    :return:
    """
    if isinstance(value, Enum):
        return str(value)
    else:
        return value


