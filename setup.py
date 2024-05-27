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
from typing import List

import setuptools

with open("README.md", "r", encoding="utf-8") as readme_file:
    LONG_DESCRIPTION = readme_file.read()


def get_requirements(filename: str) -> List[str]:
    """
    Get the required packages from the `filename` specified.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(package_dir, "requirements", filename)
    required_packages: List[str] = []
    with open(filepath, "r", encoding="utf-8") as requirements_file:
        for line in requirements_file:
            requirement = line.strip()
            if requirement and not requirement.startswith(("#", "-f")):
                required_packages.append(requirement)
    return required_packages


with open("geti_sdk/__init__.py", "r", encoding="utf-8") as init_file:
    for line in init_file:
        line = line.strip()
        if line.startswith("__version__"):
            VERSION = line.split("=")[1].strip().strip('"')

setuptools.setup(
    name="geti-sdk",
    version=VERSION,
    author="Intel OpenVINO",
    author_email="ludo.cornelissen@intel.com",
    description="Software Development Kit for the Intel® Geti™ platform",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Copyright (C) 2022 Intel Corporation - All Rights Reserved. Licensed "
    "under the Apache License, Version 2.0 (the 'License'). See LICENSE file for "
    "more details.",
    url="https://github.com/openvinotoolkit/geti-sdk",
    project_urls={
        "Documentation": "https://openvinotoolkit.github.io/geti-sdk",
        "Bug Tracker": "https://github.com/openvinotoolkit/geti-sdk/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "dev": get_requirements("requirements-dev.txt"),
        "docs": get_requirements("requirements-docs.txt"),
        "notebooks": get_requirements("requirements-notebooks.txt"),
    },
    include_package_data=True,
)
