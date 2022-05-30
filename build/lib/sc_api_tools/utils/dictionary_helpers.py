from typing import TypeVar, Dict, Any, Optional

KeyType = TypeVar("KeyType")


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
