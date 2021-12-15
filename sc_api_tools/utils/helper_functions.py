from typing import List, Optional, Any, TypeVar, Dict

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
    if isinstance(workspaces, list):
        workspace_list = workspaces
    elif isinstance(workspaces, dict):
        workspace_list = workspaces["workspaces"]
    else:
        raise ValueError(
            f"Unexpected response from cluster: {workspaces}. Expected to receive a "
            f"dictionary containing workspace data."
        )
    default_workspace = next(
        (workspace for workspace in workspace_list
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


def generate_classification_labels(
        labels: List[str], multilabel: bool = False
) -> List[Dict[str, str]]:
    """
    Generates label creation data from a list of label names. If `multiclass = True`,
    the labels will be generated in such a way that multiple labels can be assigned to
    a single image. If `multiclass = False`, only a single label per image can be
    assigned.

    :param labels: Label names to be used
    :param multilabel: True to be able to assign multiple labels to one image, False
        otherwise. Defaults to False.
    :return: List of dictionaries containing the label data that can be sent to the SC
        project creation endpoint
    """
    label_list: List[Dict[str, str]] = []
    if multilabel or len(labels) == 1:
        for label in labels:
            label_list.append({"name": label, "group": f"{label}_group"})
    else:
        for label in labels:
            label_list.append({"name": label, "group": f"default_classification_group"})
    return label_list


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
