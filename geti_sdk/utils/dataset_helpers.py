# Copyright (C) 2023 Intel Corporation
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
from typing import List

from geti_sdk.data_models import Project
from geti_sdk.data_models.project import Dataset
from geti_sdk.http_session import GetiSession
from geti_sdk.utils import deserialize_dictionary


def refresh_datasets(
    session: GetiSession, project: Project, workspace_id: str
) -> List[Dataset]:
    """
    Query the Intel® Geti™ server addressed by the `session` to retrieve the
    up to date list of datasets in the project.

    Note: the `project.datasets` attribute is updated in place with the list of
        datasets retrieved from the server

    :param session: GetiSession to the Intel® Geti™ server on which the project lives
    :param project: Project for which to refresh the datasets
    :param workspace_id: ID of the workspace in which the project lives
    :return: List of current datasets in the project
    """
    if project.id is None:
        raise ValueError(
            "Project ID is not defined, please make sure that the project you pass "
            "contains a valid ID"
        )
    response = session.get_rest_response(
        url=f"workspaces/{workspace_id}/projects/{project.id}/datasets", method="GET"
    )
    datasets = [
        deserialize_dictionary(dataset_dict, Dataset)
        for dataset_dict in response["datasets"]
    ]
    project.datasets = datasets
    return datasets


def create_dataset(
    session: GetiSession, project: Project, workspace_id: str, name: str
) -> Dataset:
    """
    Create a new dataset named `name` in the `project`.

    :param session: GetiSession to the Intel® Geti™ server on which the project lives
    :param project: Project in which to create the dataset
    :param workspace_id: ID of the workspace in which the project lives
    :param name: Name of the dataset to create
    :return: The created dataset
    """
    request_data = {"name": name, "use_for_training": False}
    response = session.get_rest_response(
        url=f"workspaces/{workspace_id}/projects/{project.id}/datasets",
        method="POST",
        data=request_data,
    )
    dataset = deserialize_dictionary(response, output_type=Dataset)
    project.datasets.append(dataset)
    return dataset
