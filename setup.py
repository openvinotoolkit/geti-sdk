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

import setuptools
from typing import List

with open("README.md", "r", encoding="utf-8") as readme_file:
    LONG_DESCRIPTION = readme_file.read()


def get_requirements(filename: str) -> List[str]:
    """
    Gets the required packages from the `filename` specified
    """
    required_packages: List[str] = []
    with open(filename, 'r', encoding="utf-8") as requirements_file:
        for line in requirements_file:
            requirement = line.strip()
            if requirement and not requirement.startswith(('#', '-f')):
                required_packages.append(requirement)
    return required_packages


with open("sc_api_tools/__init__.py", 'r', encoding="utf-8") as init_file:
    for line in init_file:
        line = line.strip()
        if line.startswith("__version__"):
            VERSION = line.split("=")[1].strip().strip("'")

setuptools.setup(
    name="sc-api-tools",
    version=VERSION,
    author="Ludo Cornelissen",
    author_email="ludo.cornelissen@intel.com",
    description="A module for interacting with the Sonoma Creek REST API",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools",
    project_urls={
        "Bug Tracker": "https://github.com/intel-innersource/frameworks.ai.interactive-ai-workflow.sonoma-creek-api-tools/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=get_requirements('requirements.txt'),
    extras_require={
        'deployment': get_requirements('requirements-deployment.txt'),
        'docs': get_requirements('./docs/requirements.txt'),
        'tests': get_requirements('./tests/requirements.txt')
    }
)
