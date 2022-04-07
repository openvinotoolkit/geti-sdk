from typing import List

import pytest
from _pytest.fixtures import FixtureRequest

from sc_api_tools import SCRESTClient
from sc_api_tools.annotation_readers import DatumAnnotationReader
from sc_api_tools.rest_managers import ProjectManager


class TestSCRESTClient:
    def test_client_initialization(self, fxt_client: SCRESTClient):
        """
        Test that the SCRESTClient is initialized properly, by checking that it obtains a
        workspace ID
        """
        assert fxt_client.workspace_id is not None

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "project_type, dataset_filter_criterion",
        [
            ("classification", "XOR"),
            ("detection", "OR"),
            ("segmentation", "OR")
        ],
        ids=["Classification project", "Detection project", "Segmentation project"]
    )
    def test_create_single_task_project_from_dataset(
        self,
        project_type,
        dataset_filter_criterion,
        fxt_annotation_reader: DatumAnnotationReader,
        fxt_client: SCRESTClient,
        fxt_default_labels: List[str],
        fxt_image_folder: str,
        fxt_project_manager: ProjectManager,
        request: FixtureRequest
    ):
        """
        Test that creating a single task project from a datumaro dataset works

        Tests project creation for classification, detection and segmentation type
        projects
        """
        project_name = f"sdk_test_{project_type}_project_from_dataset"
        fxt_annotation_reader.filter_dataset(
            labels=fxt_default_labels, criterion=dataset_filter_criterion
        )
        fxt_client.create_single_task_project_from_dataset(
            project_name=project_name,
            project_type=project_type,
            path_to_images=fxt_image_folder,
            annotation_reader=fxt_annotation_reader,
            enable_auto_train=False
        )

        def project_finalizer():
            try:
                fxt_project_manager.delete_project(
                    project=project_name, requires_confirmation=False
                )
            except TypeError as error:
                print(f"Project {project_name} was not found on the server")
                raise error
            except ValueError:
                print(
                    f"Unable to delete project '{project_name}' from the server, it "
                    f"is most likely locked for deletion due to an operation/training "
                    f"session that is in progress."
                )
        request.addfinalizer(project_finalizer)
