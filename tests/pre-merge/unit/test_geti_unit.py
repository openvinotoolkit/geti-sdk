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

import pytest
from pytest_mock import MockerFixture

from geti_sdk import Geti
from geti_sdk.data_models import Project
from geti_sdk.http_session import ServerCredentialConfig, ServerTokenConfig
from tests.helpers.constants import DUMMY_HOST, DUMMY_PASSWORD, DUMMY_TOKEN, DUMMY_USER


class TestGeti:
    def test_exception_flow_init(
        self,
        mocker: MockerFixture,
        fxt_mocked_session_factory,
        fxt_mocked_server_credential_config: ServerCredentialConfig,
    ):
        # Arrange
        mocker.patch("geti_sdk.geti.GetiSession", new=fxt_mocked_session_factory)
        mock_get_workspace_id = mocker.patch(
            "geti_sdk.geti.get_default_workspace_id", return_value=1
        )

        # Act and assert
        # host is None and server_config is None
        with pytest.raises(TypeError):
            Geti()

        # host is not None, missing credentials or token
        with pytest.raises(TypeError):
            Geti(host=DUMMY_HOST)

        # Both token and credentials specified, credentials will be ignored
        with pytest.warns():
            geti = Geti(
                host=DUMMY_HOST,
                token=DUMMY_TOKEN,
                username=DUMMY_USER,
                password=DUMMY_PASSWORD,
            )
        assert isinstance(geti.session.config, ServerTokenConfig)
        mock_get_workspace_id.assert_called_once()

        # Both host and server_config specified, host will be ignored
        with pytest.warns():
            Geti(host=DUMMY_HOST, server_config=fxt_mocked_server_credential_config)

        with pytest.warns():
            Geti(
                server_config=fxt_mocked_server_credential_config,
                proxies={"https": "http://dummy_proxy.com"},
            )

        # When the new authentication mechanism is detected (Geti v1.15 and up), do
        # not acquire token
        geti = Geti(host=DUMMY_HOST, token=DUMMY_TOKEN)
        assert "x-api-key" in geti.session.headers.keys()

    def test_logout(self, mocker: MockerFixture, fxt_mocked_geti: Geti):
        # Arrange
        mock_logout = mocker.patch.object(fxt_mocked_geti.session, "logout")

        # Act
        fxt_mocked_geti.logout()

        # Assert
        mock_logout.assert_called_once()

    def test_download_all_projects(
        self,
        mocker: MockerFixture,
        fxt_mocked_geti: Geti,
        fxt_temp_directory: str,
        fxt_nightly_projects: List[Project],
    ):
        # Arrange
        mock_get_all_projects = mocker.patch(
            "geti_sdk.geti.ProjectClient.get_all_projects",
            return_value=fxt_nightly_projects,
        )
        mock_download_project_data = mocker.patch(
            "geti_sdk.import_export.import_export_module.GetiIE.download_project_data"
        )

        # Act
        projects = fxt_mocked_geti.download_all_projects(
            target_folder=fxt_temp_directory, include_predictions=False
        )

        # Assert
        mock_get_all_projects.assert_called_once()
        assert mock_download_project_data.call_count == len(projects)

    def test_upload_all_projects(
        self,
        mocker: MockerFixture,
        fxt_mocked_geti: Geti,
        fxt_temp_directory: str,
        fxt_nightly_projects: List[Project],
    ):
        # Arrange
        target_dir = os.path.join(fxt_temp_directory, "test_upload_all_projects")
        for project in fxt_nightly_projects:
            os.makedirs(os.path.join(target_dir, project.name))
        mock_is_project_dir = mocker.patch(
            "geti_sdk.geti.ProjectClient.is_project_dir", return_value=True
        )
        mock_upload_project_data = mocker.patch(
            "geti_sdk.import_export.import_export_module.GetiIE.upload_project_data"
        )

        # Act
        fxt_mocked_geti.upload_all_projects(target_folder=target_dir)

        # Assert
        assert mock_is_project_dir.call_count == len(fxt_nightly_projects)
        assert mock_upload_project_data.call_count == len(fxt_nightly_projects)
