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

import os
import re
from importlib import resources
from pathlib import Path
from typing import Tuple

from pathvalidate import sanitize_filepath

from geti_sdk.data_models import OptimizedModel, Project

try:
    OVMS_README_PATH = str(
        resources.files("geti_sdk.deployment.resources") / "OVMS_README.md"
    )
except AttributeError:
    with resources.path("geti_sdk.deployment.resources", "OVMS_README.md") as data_path:
        OVMS_README_PATH = str(data_path)


def generate_ovms_model_name(
    project: Project, model: OptimizedModel, omit_version: bool = True
) -> str:
    """
    Generate a valid name for a model to be uploaded to the OpenVINO Model Server.

    :param project: Project the model belongs to
    :param model: The model for which to generate the name
    :param omit_version: True not to include the version of the model in the name.
        This is useful if the name needs to be used to create a directory structure,
        since including the version will break this.
    :return: String containing the name of the model
    """
    model_name = sanitize_filepath(project.name + "_" + model.name)
    model_name = model_name.replace(" ", "_")
    model_name = model_name.replace("-", "_")
    model_name = model_name.lower()
    if model.version is not None and not omit_version:
        model_name += f":{model.version}"
    return model_name


def generate_ovms_model_address(ovms_address: str, model_name: str) -> str:
    """
    Generate a string representing a valid model address for the OpenVINO Model Server,
    from a given `ovms_address` and `model_name`.

    The format expected for model addresses by OVMS is:

        <address>:<port>/models/<model_name>[:<model_version>]

    Where `address` must not contain the protocol (http or https)

    :param ovms_address: IP address or URL pointing to the OVMS instance, including
        port specification
    :param model_name: Name of the model
    :return: String containing the model address
    """
    model_address = f"{ovms_address}/models/{model_name}"
    if model_address.startswith("https://"):
        model_address = model_address[8:]
    if model_address.startswith("http://"):
        model_address = model_address[7:]
    return model_address


def target_device_is_ovms(device: str) -> bool:
    """
    Return True if the target `device` specified is a URL or IP address, False otherwise

    :param device: Target device string to check
    :return: True if the string represents a URL or IP address, False otherwise
    """
    # Check if 'device' has been specified as a URL or IP address.
    server_pattern = re.compile(
        r"^((https?://)|(www.))(?:([a-zA-Z]+)|(\d+\.\d+\.\d+\.\d+)):\d{1,5}?$"
    )
    return server_pattern.match(device) is not None


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert an RGB color value to its corresponding hexadecimal representation.

    :param rgb: A tuple representing the RGB color value, where each element is an integer between 0 and 255.
    :return: The hexadecimal representation of the RGB color value.

    _Example:

        >>> rgb_to_hex((255, 0, 0))
        '#ff0000'
    """
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def get_package_version_from_requirements(
    requirements_path: os.PathLike, package_name: str
) -> str:
    """
    Get the version of a package from a requirements file.

    :param requirements_path: The requirements file path
    :param package_name: The name of the package to get the version of
    :return: The version of the package as a string, empty line if the package is not found.
    :raises: ValueError If the requirements file is not found.
    """
    if not package_name:
        return ""

    requirements_path = Path(requirements_path).resolve()
    if not requirements_path.exists():
        raise ValueError(f"Requirements file {requirements_path} not found")

    for line in requirements_path.read_text().split("\n"):
        if line.startswith(package_name):
            return line.split("==")[1].strip()
    return ""
