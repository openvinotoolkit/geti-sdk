# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import glob
import os
from typing import Any

import yaml

from .constants import DUMMY_HOST, CASSETTE_PATH, CASSETTE_EXTENSION


def are_cassettes_available() -> bool:
    """
    Checks that the VCR cassettes required to run the tests offline are available

    :return: True if the cassettes are available in the proper path, False otherwise
    """
    if not os.path.isdir(CASSETTE_PATH):
        return False
    if len(os.listdir(CASSETTE_PATH)) > 0:
        return True
    return False


def replace_host_name_in_cassettes(server_address: str) -> None:
    """
    This function searches for the server_address in all cassette files and
    replaces all occurrences of that address by 'dummy_host'. The cassette files are
    updated in place

    :param server_address: Server address to search for and replace
    """
    host_name = server_address.replace("https://", "").strip("/")
    for cassette_path in glob.glob(
            os.path.join(CASSETTE_PATH, f"*.{CASSETTE_EXTENSION}")
    ):
        replace_host_name_in_cassette(
            host_name=host_name, path_to_cassette_file=cassette_path
        )


def replace_host_name_in_cassette(
        host_name: str, path_to_cassette_file: str
) -> None:
    """
    This function searches for the host_name in a target cassette file and
    replaces all occurrences of that address by 'dummy_host'. The cassette file is
    updated in place

    :param host_name: Host name to search for and replace
    :param path_to_cassette_file: Path to the cassette file to search in
    """
    with open(path_to_cassette_file, 'r') as cassette_file:
        cassette_dict = yaml.safe_load(cassette_file)
    scrubbed_dict = __replace_string_in_input(
        cassette_dict, to_replace=host_name, replace_with=DUMMY_HOST
    )
    os.remove(path_to_cassette_file)
    with open(path_to_cassette_file, 'w') as new_cassette_file:
        yaml.dump(scrubbed_dict, new_cassette_file)


def __replace_string_in_input(
        input_dict: Any, to_replace: str, replace_with: str
) -> Any:
    """
    Replaces the string `to_replace` with `replace_with`, in all strings found inside
    the input dictionary

    :param input_dict: Dictionary to replace strings in
    :param to_replace: Target string or substring to replace
    :param replace_with: String to replace the target string with
    :return: Updated dictionary with all occurrences of the string `to_replace`
        replaced by the string in `replace_with`
    """
    if isinstance(input_dict, dict):
        for key, value in input_dict.items():
            if isinstance(value, str):
                input_dict[key] = value.replace(to_replace, replace_with)
            else:
                input_dict[key] = __replace_string_in_input(
                    value, to_replace, replace_with
                )
    elif isinstance(input_dict, list):
        new_list = []
        for item in input_dict:
            new_list.append(__replace_string_in_input(
                item, to_replace, replace_with)
            )
        return new_list
    elif isinstance(input_dict, str):
        return input_dict.replace(to_replace, replace_with)
    return input_dict
