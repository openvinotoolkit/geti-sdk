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

import pytest

from geti_sdk.geti import Geti


class TestImportExport:
    """
    Integration tests for the Import Export methods in the Geti class.
    """

    @pytest.mark.vcr()
    def test_export_import_project(
        self,
        fxt_geti: Geti,
        fxt_temp_directory: str,
    ) -> None:
        project = fxt_geti.project_client.get_all_projects(get_project_details=False)[0]
        target_folder = os.path.join(fxt_temp_directory, project.name + "_snapshot")
        archive_path = target_folder + "/project_archive.zip"
        imported_project_name = "IMPORTED_PROJECT"

        # Project is exported
        assert not os.path.exists(archive_path)
        fxt_geti.export_project(
            project_name=project.name, project_id=project.id, filepath=archive_path
        )
        assert os.path.exists(archive_path)
        # Project is imported
        existing_projects_pre_import = fxt_geti.project_client.get_all_projects(
            get_project_details=False
        )
        imported_project = fxt_geti.import_project(
            filepath=archive_path, project_name=imported_project_name
        )
        assert imported_project.name == imported_project_name
        existing_projects = fxt_geti.project_client.get_all_projects(
            get_project_details=False
        )
        assert (
            next((p for p in existing_projects if p.id == imported_project.id), None)
            is not None
        )
        assert (
            next(
                (
                    p
                    for p in existing_projects_pre_import
                    if p.id == imported_project.id
                ),
                None,
            )
            is None
        )
        # Project is deleted
        fxt_geti.project_client.delete_project(
            imported_project, requires_confirmation=False
        )

    @pytest.mark.vcr()
    def test_export_import_dataset(
        self,
        fxt_geti: Geti,
        fxt_temp_directory: str,
    ) -> None:
        project = fxt_geti.project_client.get_all_projects(get_project_details=False)[0]
        project = fxt_geti.project_client._get_project_detail(project)
        assert project.datasets
        dataset = project.datasets[0]
        target_folder = os.path.join(fxt_temp_directory, project.name + "_snapshot")
        archive_path = target_folder + "/dataset_archive.zip"
        imported_project_name = "IMPORTED_PROJECT_FROM_DATASET"

        # Dataset is exported
        assert not os.path.exists(archive_path)
        fxt_geti.export_dataset(project=project, dataset=dataset, filepath=archive_path)
        assert os.path.exists(archive_path)
        # Dataset is imported as a project
        existing_projects_pre_import = fxt_geti.project_client.get_all_projects(
            get_project_details=False
        )
        imported_project = fxt_geti.import_dataset(
            filepath=archive_path,
            project_name=imported_project_name,
            project_type=project.project_type,
        )
        assert imported_project.name == imported_project_name
        existing_projects = fxt_geti.project_client.get_all_projects(
            get_project_details=False
        )
        assert (
            next((p for p in existing_projects if p.id == imported_project.id), None)
            is not None
        )
        assert (
            next(
                (
                    p
                    for p in existing_projects_pre_import
                    if p.id == imported_project.id
                ),
                None,
            )
            is None
        )
        # Project is deleted
        fxt_geti.project_client.delete_project(
            imported_project, requires_confirmation=False
        )
