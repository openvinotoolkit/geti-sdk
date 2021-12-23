from datetime import datetime
from enum import Enum

import attr
from typing import Any, Union, Optional

from sc_api_tools.data_models import TaskType
from sc_api_tools.data_models.enums import ShapeType, MediaType


def deidentify(instance: Any):
    """
    Sets all identifier fields of an instance of an attr.s decorated class within the
    SC REST DataModels to None

    :param instance: Object to deidentify
    """
    for field in attr.fields(type(instance)):
        if field.name in instance._identifier_fields:
            setattr(instance, field.name, None)


def str_to_task_type(task_type: Union[str, TaskType]) -> TaskType:
    """
    Converts an input string to a task type

    :param task_type:
    :return: TaskType instance corresponding to `task_type`
    """
    if isinstance(task_type, str):
        return TaskType(task_type)
    else:
        return task_type


def str_to_media_type(media_type: Union[str, MediaType]) -> MediaType:
    """
    Converts an input string to a media type

    :param media_type:
    :return: MediaType instance corresponding to `media_type`
    """
    if isinstance(media_type, str):
        return MediaType(media_type)
    else:
        return media_type


def str_to_shape_type(shape_type: Union[str, ShapeType]) -> ShapeType:
    """
    Converts an input string to a shape type

    :param shape_type:
    :return: ShapeType instance corresponding to `shape_type`
    """
    if isinstance(shape_type, str):
        return ShapeType(shape_type)
    else:
        return shape_type


def str_to_datetime(datetime_str: Optional[Union[str, datetime]]) -> Optional[datetime]:
    """
    Converts a string to a datetime

    :param datetime_str:
    :return: datetime instance
    """
    if isinstance(datetime_str, str):
        return datetime.fromisoformat(datetime_str)
    elif isinstance(datetime_str, datetime):
        return datetime_str
    elif datetime_str is None:
        return None


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


def attr_value_serializer(instance, field, value):
    """
    Converts a value in an attr.s decorated class to string representation, used
    while converting the attrs object to a dictionary.

    Converts Enums and datetime objects to string representation

    :param instance:
    :param field:
    :param value:
    :return:
    """
    if isinstance(value, Enum):
        return str(value)
    elif isinstance(value, datetime):
        return datetime.isoformat(value)
    else:
        return value
