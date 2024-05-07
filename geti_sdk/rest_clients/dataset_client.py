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
import os
from typing import List, Optional

from geti_sdk.data_models import Dataset, Project
from geti_sdk.http_session import GetiSession
from geti_sdk.utils import deserialize_dictionary


class DatasetClient:
    """
    Class to manage datasets for a certain Intel® Geti™ project.
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/projects/{project.id}/datasets"

    def create_dataset(self, name: str) -> Dataset:
        """
        Create a new dataset named `name` inside the project.

        :param name: Name of the dataset to create
        :return: The created dataset
        """
        request_data = {"name": name}
        response = self.session.get_rest_response(
            url=self.base_url,
            method="POST",
            data=request_data,
        )
        dataset = deserialize_dictionary(response, output_type=Dataset)
        self.project.datasets.append(dataset)
        return dataset

    def get_all_datasets(self) -> List[Dataset]:
        """
        Query the Intel® Geti™ server to retrieve an up to date list of datasets in
        the project.

        :return: List of current datasets in the project
        """
        if self.project.id is None:
            raise ValueError(
                "Project ID is not defined, please make sure that the project you pass "
                "contains a valid ID"
            )
        response = self.session.get_rest_response(url=self.base_url, method="GET")
        datasets = [
            deserialize_dictionary(dataset_dict, Dataset)
            for dataset_dict in response["datasets"]
        ]
        self.project.datasets = datasets
        return datasets

    def get_dataset_statistics(self, dataset: Dataset) -> dict:
        """
        Retrieve the media and annotation statistics for a particular dataset
        """
        response = self.session.get_rest_response(
            url=f"{self.base_url}/{dataset.id}/statistics", method="GET"
        )
        return response

    def get_dataset_by_name(self, dataset_name: str) -> Dataset:
        """
        Retrieve a dataset by name

        :param dataset_name: Name of the dataset to retrieve
        :return: Dataset object
        """
        dataset: Optional[Dataset] = None
        for ds in self.get_all_datasets():
            if ds.name == dataset_name:
                dataset = ds
        if dataset is None:
            raise ValueError(
                f"Dataset named '{dataset_name}' was not found in project "
                f"'{self.project.name}'"
            )
        return dataset

    def has_dataset_subfolders(self, path_to_folder: str) -> bool:
        """
        Check if a project folder has it's media folders organized according to the
        datasets in the project

        :param path_to_folder: Path to the project folder to check
        :return: True if the media folders in the project folder tree are organized
            according to the datasets in the project.
        """
        media_folder_names = ["images", "videos"]
        result: bool = True
        for folder_name in media_folder_names:
            full_path = os.path.join(path_to_folder, folder_name)
            if not os.path.exists(full_path):
                continue
            result *= self._media_folder_has_dataset_subfolders(full_path)
        return result

    def _media_folder_has_dataset_subfolders(self, path_to_folder: str) -> bool:
        """
        Check if a folder with media has subfolders organized according to the
        projects' datasets

        :param path_to_folder: Path to the media folder to check for dataset subfolders
        :return: True if the folder contains subfolders named according to the
            datasets in the project, False otherwise
        """
        if not os.path.isdir(path_to_folder):
            return False
        content = os.listdir(path_to_folder)
        for dataset in self.project.datasets:
            if dataset.name not in content:
                return False
            if not os.path.isdir(os.path.join(path_to_folder, dataset.name)):
                return False
        return True
