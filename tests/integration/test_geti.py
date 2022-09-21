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
import logging
import os
import time
from typing import List

import cv2
import pytest
from _pytest.fixtures import FixtureRequest
from vcr import VCR

from geti_sdk import Geti
from geti_sdk.annotation_readers import DatumAnnotationReader
from geti_sdk.data_models import Prediction, Project
from geti_sdk.http_session import GetiRequestException
from geti_sdk.rest_clients import AnnotationClient, ImageClient
from tests.helpers import (
    ProjectService,
    SdkTestMode,
    attempt_to_train_task,
    get_or_create_annotated_project_for_test_class,
)
from tests.helpers.constants import CASSETTE_EXTENSION, PROJECT_PREFIX


class TestGeti:
    """
    Integration tests for the methods in the Geti class.

    NOTE: These tests are meant to be run in one go
    """

    @staticmethod
    def ensure_annotated_project(
        project_service: ProjectService, annotation_reader: DatumAnnotationReader
    ) -> Project:
        return get_or_create_annotated_project_for_test_class(
            project_service=project_service,
            annotation_readers=[annotation_reader],
            project_type="detection",
            project_name=f"{PROJECT_PREFIX}_geti",
            enable_auto_train=False,
        )

    def test_project_setup(
        self,
        fxt_project_service: ProjectService,
        fxt_annotation_reader: DatumAnnotationReader,
        fxt_vcr: VCR,
        fxt_test_mode: SdkTestMode,
    ):
        """
        This test sets up an annotated project on the server, that persists for the
        duration of this test class. The project will train while the project
        creation tests are running.
        """
        self.ensure_annotated_project(fxt_project_service, fxt_annotation_reader)
        assert fxt_project_service.has_project

        # For the integration tests we start training manually
        with fxt_vcr.use_cassette(
            f"{fxt_project_service.project.name}_setup_training.{CASSETTE_EXTENSION}"
        ):
            attempt_to_train_task(
                training_client=fxt_project_service.training_client,
                task=0,
                test_mode=fxt_test_mode,
            )

        # Wait a few secs to check whether the project is training
        if fxt_test_mode != SdkTestMode.OFFLINE:
            time.sleep(5)

        assert fxt_project_service.is_training

    def test_geti_initialization(self, fxt_geti: Geti):
        """
        Test that the Geti instance is initialized properly, by checking that it
        obtains a workspace ID
        """
        assert fxt_geti.workspace_id is not None

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "project_type, dataset_filter_criterion",
        [
            ("classification", "XOR"),
            ("detection", "OR"),
            ("segmentation", "OR"),
            ("instance_segmentation", "OR"),
            ("rotated_detection", "OR"),
        ],
        ids=[
            "Classification project",
            "Detection project",
            "Segmentation project",
            "Instance segmentation project",
            "Rotated detection project",
        ],
    )
    def test_create_single_task_project_from_dataset(
        self,
        project_type,
        dataset_filter_criterion,
        fxt_annotation_reader: DatumAnnotationReader,
        fxt_geti: Geti,
        fxt_default_labels: List[str],
        fxt_image_folder: str,
        fxt_project_finalizer,
        request: FixtureRequest,
    ):
        """
        Test that creating a single task project from a datumaro dataset works

        Tests project creation for classification, detection and segmentation type
        projects
        """
        project_name = f"{PROJECT_PREFIX}_{project_type}_project_from_dataset"
        fxt_annotation_reader.filter_dataset(
            labels=fxt_default_labels, criterion=dataset_filter_criterion
        )
        fxt_geti.create_single_task_project_from_dataset(
            project_name=project_name,
            project_type=project_type,
            path_to_images=fxt_image_folder,
            annotation_reader=fxt_annotation_reader,
            enable_auto_train=False,
        )

        request.addfinalizer(lambda: fxt_project_finalizer(project_name))

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "project_type",
        ["detection_to_classification", "detection_to_segmentation"],
        ids=[
            "Detection to classification project",
            "Detection to segmentation project",
        ],
    )
    def test_create_task_chain_project_from_dataset(
        self,
        project_type,
        fxt_annotation_reader_factory,
        fxt_geti: Geti,
        fxt_default_labels: List[str],
        fxt_image_folder: str,
        fxt_project_finalizer,
        request: FixtureRequest,
    ):
        """
        Test that creating a task chain project from a datumaro dataset works

        Tests project creation for:
          detection -> classification
          detection -> segmentation
        """
        project_name = f"{PROJECT_PREFIX}_{project_type}_project_from_dataset"
        annotation_reader_task_1 = fxt_annotation_reader_factory()
        annotation_reader_task_2 = fxt_annotation_reader_factory()
        annotation_reader_task_1.filter_dataset(
            labels=fxt_default_labels, criterion="OR"
        )
        annotation_reader_task_2.filter_dataset(
            labels=fxt_default_labels, criterion="OR"
        )
        annotation_reader_task_1.group_labels(
            labels_to_group=fxt_default_labels, group_name="block"
        )
        project = fxt_geti.create_task_chain_project_from_dataset(
            project_name=project_name,
            project_type=project_type,
            path_to_images=fxt_image_folder,
            label_source_per_task=[annotation_reader_task_1, annotation_reader_task_2],
            enable_auto_train=False,
        )
        request.addfinalizer(lambda: fxt_project_finalizer(project_name))

        all_labels = fxt_default_labels + ["block"]
        for label_name in all_labels:
            assert label_name in [label.name for label in project.get_all_labels()]

    @pytest.mark.vcr()
    def test_download_and_upload_project(
        self,
        fxt_project_service: ProjectService,
        fxt_geti: Geti,
        fxt_temp_directory: str,
        fxt_project_finalizer,
        request: FixtureRequest,
    ):
        """
        Test that downloading a project works as expected.

        :param fxt_project_service:
        :return:
        """
        project = fxt_project_service.project
        target_folder = os.path.join(fxt_temp_directory, project.name)

        fxt_geti.download_project(project.name, target_folder=target_folder)

        assert os.path.isdir(target_folder)
        assert "project.json" in os.listdir(target_folder)

        n_images = len(os.listdir(os.path.join(target_folder, "images")))
        n_annotations = len(os.listdir(os.path.join(target_folder, "annotations")))

        uploaded_project = fxt_geti.upload_project(
            target_folder=target_folder,
            project_name=f"{PROJECT_PREFIX}_upload",
            enable_auto_train=False,
        )
        request.addfinalizer(lambda: fxt_project_finalizer(uploaded_project.name))
        image_client = ImageClient(
            session=fxt_geti.session,
            workspace_id=fxt_geti.workspace_id,
            project=uploaded_project,
        )
        images = image_client.get_all_images()
        assert len(images) == n_images

        annotation_client = AnnotationClient(
            session=fxt_geti.session,
            workspace_id=fxt_geti.workspace_id,
            project=uploaded_project,
        )
        annotation_target_folder = os.path.join(
            fxt_temp_directory, "uploaded_annotations"
        )
        annotation_client.download_annotations_for_images(
            images, annotation_target_folder
        )
        assert (
            len(os.listdir(os.path.join(annotation_target_folder, "annotations")))
            == n_annotations
        )

    @pytest.mark.vcr()
    def test_upload_and_predict_image(
        self,
        fxt_geti: Geti,
        fxt_image_path: str,
        fxt_project_service: ProjectService,
        fxt_test_mode: SdkTestMode,
    ) -> None:
        """
        Verifies that the upload_and_predict_image method works correctly
        """
        project = fxt_project_service.project
        # If training is not ready yet, monitor progress until job completes
        if not fxt_project_service.prediction_client.ready_to_predict:
            timeout = 300 if fxt_test_mode != SdkTestMode.OFFLINE else 1
            jobs = fxt_project_service.training_client.get_jobs(project_only=True)
            fxt_project_service.training_client.monitor_jobs(jobs, timeout=timeout)

        # Make several attempts to get the prediction, first attempts trigger the
        # inference server to start up but the requests may time out
        n_attempts = 2 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        sleep_time = 20 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        for j in range(n_attempts):
            try:
                image, prediction = fxt_geti.upload_and_predict_image(
                    project_name=project.name,
                    image=fxt_image_path,
                    visualise_output=False,
                    delete_after_prediction=False,
                )
            except GetiRequestException as error:
                prediction = None
                time.sleep(sleep_time)
                logging.info(error)
            if prediction is not None:
                assert isinstance(prediction, Prediction)
                break

    @pytest.mark.vcr()
    def test_deployment(
        self,
        fxt_geti: Geti,
        fxt_image_path: str,
        fxt_project_service: ProjectService,
        fxt_temp_directory: str,
    ) -> None:
        """
        Verifies that deploying a project works
        """
        project = fxt_project_service.project
        deployment = fxt_geti.deploy_project(
            project.name, output_folder=fxt_temp_directory
        )
        deployment_folder = os.path.join(fxt_temp_directory, project.name)
        assert os.path.isdir(deployment_folder)
        deployment.load_inference_models(device="CPU")

        image_bgr = cv2.imread(fxt_image_path)
        image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        local_prediction = deployment.infer(image_np)
        assert isinstance(local_prediction, Prediction)
        image, online_prediction = fxt_geti.upload_and_predict_image(
            project.name,
            image=image_np,
            delete_after_prediction=True,
            visualise_output=False,
        )

        online_mask = online_prediction.as_mask(image.media_information)
        local_mask = local_prediction.as_mask(image.media_information)

        assert online_mask.shape == local_mask.shape
        # assert np.all(local_mask == online_mask)
