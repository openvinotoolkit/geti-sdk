from geti_sdk import Geti


class TestGeti:
    def test_download_all_projects(
        self, mocker, fxt_mocked_geti: Geti, fxt_temp_directory: str
    ):
        # Arrange
        mocker.patch("geti_sdk.geti.ProjectClient")
        mocker.patch(fxt_mocked_geti, "download_project")

        assert fxt_mocked_geti.workspace_id == 1
