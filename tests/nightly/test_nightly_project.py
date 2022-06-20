import os
import time
from typing import ClassVar, List

import cv2
import numpy as np

from sc_api_tools import SCRESTClient
from sc_api_tools.annotation_readers import DatumAnnotationReader
from sc_api_tools.data_models import Prediction, Job
from sc_api_tools.data_models.enums import JobState
from sc_api_tools.http_session import SCRequestException
from tests.helpers import (
    ProjectService,
    get_or_create_annotated_project_for_test_class,
    plot_predictions_side_by_side
)
from tests.helpers.constants import PROJECT_PREFIX


class TestNightlyProject:
    PROJECT_TYPE: ClassVar[str] = "none"

    # Setting __test__ to False indicates to pytest that this class is not part of
    # the tests. This allows it to be imported in other test files.
    __test__: ClassVar[bool] = False

    def test_project_setup(
            self,
            fxt_project_service_no_vcr: ProjectService,
            fxt_annotation_reader: DatumAnnotationReader,
            fxt_annotation_reader_grouped: DatumAnnotationReader,
            fxt_learning_parameter_settings: str
    ):
        """
        This test sets up an annotated project on the server, that persists for the
        duration of this test class.
        """
        if self.PROJECT_TYPE == "classification":
            fxt_annotation_reader.filter_dataset(
                labels=["cube", "cylinder"], criterion="XOR"
            )

        annotation_readers = [fxt_annotation_reader]
        if "_to_" in self.PROJECT_TYPE:
            annotation_readers = [fxt_annotation_reader_grouped, fxt_annotation_reader]

        get_or_create_annotated_project_for_test_class(
            project_service=fxt_project_service_no_vcr,
            annotation_readers=annotation_readers,
            project_type=self.PROJECT_TYPE,
            project_name=f"{PROJECT_PREFIX}_nightly_{self.PROJECT_TYPE}",
            enable_auto_train=True,
            learning_parameter_settings=fxt_learning_parameter_settings
        )

    def test_monitor_jobs(self, fxt_project_service_no_vcr: ProjectService):
        """
        This test monitors training jobs for the project, and completes when the jobs
        are finished
        """
        training_manager = fxt_project_service_no_vcr.training_manager
        max_attempts = 3
        jobs: List[Job] = []
        n = 0
        while len(jobs) == 0 and n < max_attempts:
            jobs = training_manager.get_jobs(project_only=True)
            n += 1
            # If no jobs are found yet, wait for a while and retry
            time.sleep(10)

        if len(jobs) == 0 and n == max_attempts:
            raise RuntimeError(
                f"No auto-train job was started on the platform for project "
                f"'{fxt_project_service_no_vcr.project.name}'. Test failed."
            )

        jobs = training_manager.monitor_jobs(jobs=jobs, timeout=10000)
        for job in jobs:
            assert job.status.state == JobState.FINISHED

    def test_upload_and_predict_image(
            self,
            fxt_project_service_no_vcr: ProjectService,
            fxt_image_path: str,
            fxt_client_no_vcr: SCRESTClient
    ):
        """
        Tests uploading and predicting an image to the project. Waits for the
        inference servers to be initialized.
        """
        n_attempts = 3
        project = fxt_project_service_no_vcr.project

        for j in range(n_attempts):
            try:
                image, prediction = fxt_client_no_vcr.upload_and_predict_image(
                    project_name=project.name,
                    image=fxt_image_path,
                    visualise_output=False,
                    delete_after_prediction=False
                )
            except SCRequestException as error:
                prediction = None
                time.sleep(20)
                print(error)
            if prediction is not None:
                assert isinstance(prediction, Prediction)
                break

    def test_deployment(
            self,
            fxt_project_service_no_vcr: ProjectService,
            fxt_client_no_vcr: SCRESTClient,
            fxt_temp_directory: str,
            fxt_image_path: str,
            fxt_image_path_complex: str,
            fxt_artifact_directory: str
    ):
        """
        Tests local deployment for the project. Compares the local prediction to the
        platform prediction for a sample image. Test passes if they are equal
        """
        project = fxt_project_service_no_vcr.project

        deployment_folder = os.path.join(fxt_temp_directory, project.name)
        deployment = fxt_client_no_vcr.deploy_project(
            project.name, output_folder=deployment_folder
        )

        assert os.path.isdir(os.path.join(deployment_folder, 'deployment'))
        deployment.load_inference_models(device="CPU")

        images = {'simple': fxt_image_path, 'complex': fxt_image_path_complex}

        for image_name, image_path in images.items():
            image_bgr = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            local_prediction = deployment.infer(image_np)
            assert isinstance(local_prediction, Prediction)
            image, online_prediction = fxt_client_no_vcr.upload_and_predict_image(
                project.name,
                image=image_bgr,
                delete_after_prediction=True,
                visualise_output=False
            )

            online_mask = online_prediction.as_mask(image.media_information)
            local_mask = local_prediction.as_mask(image.media_information)

            assert online_mask.shape == local_mask.shape
            equal_masks = np.all(local_mask == online_mask)
            if not equal_masks:
                print("WARNING: local and online prediction masks are not equal!")
                print(
                    f"Number of shapes: {len(local_prediction.annotations)} - local   "
                    f"----    {len(online_prediction.annotations)} - online."
                )

            print("\n\n-------- Local prediction --------")
            print(local_prediction.overview)
            print("\n\n-------- Online prediction --------")
            print(online_prediction.overview)

            # Save the predictions as test artifacts
            predictions_dir = os.path.join(fxt_artifact_directory, 'predictions')
            if not os.path.isdir(predictions_dir):
                os.makedirs(predictions_dir)

            image_path = os.path.join(
                predictions_dir, project.name + '_' + image_name + '.jpg'
            )
            plot_predictions_side_by_side(
                image,
                prediction_1=local_prediction,
                prediction_2=online_prediction,
                filepath=image_path
            )
