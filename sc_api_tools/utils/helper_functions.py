from typing import List

from sc_api_tools.http_session import SCSession


def get_default_workspace_id(rest_session: SCSession):
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
    return [f"{label} shape" for label in detection_labels]


def get_dict_key_from_value(input_dict: dict, value):
    keys = list(input_dict.keys())
    values = list(input_dict.values())
    try:
        index = values.index(value)
    except ValueError:
        return None
    return keys[index]


def grouped(iterable, n):
    """
    Iterates over iterable, yielding n items at a time.
    """
    return zip(*[iter(iterable)] * n)
