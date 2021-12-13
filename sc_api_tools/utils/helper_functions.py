from typing import List, Optional, Any, TypeVar, Dict, Iterable

from sc_api_tools.http_session import SCSession

KeyType = TypeVar("KeyType")


def get_default_workspace_id(rest_session: SCSession) -> str:
    """
    Returns the id of the default workspace on the cluster

    :param rest_session: HTTP session to the cluser
    :return: string containing the id of the default workspace
    """
    workspaces = rest_session.get_rest_response(
        url="workspaces",
        method="GET"
    )
    default_workspace = next(
        (workspace for workspace in workspaces
         if workspace["name"] == "Default Workspace")
    )
    return default_workspace["id"]


def generate_segmentation_labels(detection_labels: List[str]) -> List[str]:
    """
    Generate segmentation label names from a list of detection label names.

    :param detection_labels: label names to generate segmentation label names for
    :return: List of label names
    """
    return [f"{label} shape" for label in detection_labels]


def get_dict_key_from_value(
        input_dict: Dict[KeyType, Any], value
) -> Optional[KeyType]:
    """
    Returns the key associated with `value` in a dictionary `input_dict`. If the value
    is not found in the dictionary, returns None

    :param input_dict: Dictionary to search in
    :param value: value to search for
    :return: key associated with value if value is in the input_dict, None otherwise
    """
    keys = list(input_dict.keys())
    values = list(input_dict.values())
    try:
        index = values.index(value)
    except ValueError:
        return None
    return keys[index]


def grouped(iterable, n: int) -> Iterable:
    """
    Iterates over iterable, yielding n items at a time.
    """
    return zip(*[iter(iterable)] * n)


def remove_null_fields(input: Any):
    """
    Remove fields that have 'None' or an emtpy string '' as their value from a
    dictionary

    NOTE: This function modifies the input dictionary in place

    :param input: Dictionary to remove the null fields from
    """
    if isinstance(input, dict):
        for key, value in list(input.items()):
            if isinstance(value, dict):
                remove_null_fields(value)
            elif value is None or value == "":
                input.pop(key)
            elif isinstance(value, list):
                for item in value:
                    remove_null_fields(item)
    elif isinstance(input, list):
        for item in input:
            remove_null_fields(item)
