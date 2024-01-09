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

import json
import os
from importlib import resources
from typing import Optional

from geti_sdk.data_models.containers import AlgorithmList
from geti_sdk.data_models.enums import Domain, TaskType
from geti_sdk.http_session import GetiSession
from geti_sdk.platform_versions import GETI_18_VERSION

try:
    UTILS_PATH = str(resources.files("geti_sdk.utils"))
except AttributeError:
    with resources.path("geti_sdk", "utils") as data_path:
        UTILS_PATH = str(data_path)

LEGACY_ALGO_PATH = os.path.join(UTILS_PATH, "legacy_algorithms.json")


def get_supported_algorithms(
    rest_session: GetiSession,
    domain: Optional[Domain] = None,
    task_type: Optional[TaskType] = None,
) -> AlgorithmList:
    """
    Return the list of supported algorithms (including algorithm metadata) for the
    cluster.

    :param rest_session: HTTP session to the cluster
    :param domain: Optional domain for which to get the supported algorithms. If left
        as None (the default), the supported algorithms for all domains are returned.
        NOTE: domain is deprecated in SC1.1, please use `task_type` instead.
    :param task_type: Optional TaskType for which to get the supported algorithms.
    :return: AlgorithmList holding the supported algorithms
    """
    filter_by_task_type = False
    if task_type is not None and domain is not None:
        raise ValueError("Please specify either task type or domain, but not both")
    elif task_type is not None:
        query = f"?task_type={task_type}"
        filter_by_task_type = True
    elif domain is not None:
        task_type = TaskType.from_domain(domain)
        query = f"?task_type={task_type}"
        filter_by_task_type = True
    else:
        query = ""

    if rest_session.version <= GETI_18_VERSION:
        algorithm_rest_response = rest_session.get_rest_response(
            url=f"supported_algorithms{query}", method="GET"
        )
    else:
        with open(LEGACY_ALGO_PATH, "r") as f:
            algorithm_rest_response = json.load(f)
        if filter_by_task_type:
            filtered_response = [
                algo
                for algo in algorithm_rest_response["supported_algorithms"]
                if algo["task_type"].upper() == task_type.name
            ]
            algorithm_rest_response["items"] = filtered_response
    return AlgorithmList.from_rest(algorithm_rest_response)
