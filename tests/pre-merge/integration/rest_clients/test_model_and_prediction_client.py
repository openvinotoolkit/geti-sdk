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
import copy
import os
from typing import List

import pytest

from geti_sdk.annotation_readers import DatumAnnotationReader
from geti_sdk.data_models import Project, TaskType
from geti_sdk.data_models.enums import JobState, PredictionMode
from geti_sdk.utils import get_supported_algorithms
from tests.helpers import (
    ProjectService,
    SdkTestMode,
    attempt_to_train_task,
    get_or_create_annotated_project_for_test_class,
)
from tests.helpers.constants import PROJECT_PREFIX


class TestModelAndPredictionClient:
    @staticmethod
    def ensure_annotated_project(
        project_service: ProjectService, annotation_reader: DatumAnnotationReader
    ) -> Project:
        return get_or_create_annotated_project_for_test_class(
            project_service=project_service,
            annotation_readers=[annotation_reader],
            project_type="classification",
            project_name=f"{PROJECT_PREFIX}_model_and_prediction_client",
        )

    @pytest.mark.vcr()
    def test_project_setup_and_get_model_by_job(
        self,
        fxt_project_service: ProjectService,
        fxt_annotation_reader: DatumAnnotationReader,
        fxt_default_labels: List[str],
        fxt_test_mode: SdkTestMode,
    ) -> None:
        """
        Sets up the project and starts training a model as preparation for further tests

        Implements two test cases:
            1. Requesting image prediction while no models trained raises ValueError
            2. Fetching the model resulting from a training job results in valid model
        """
        annotation_reader = fxt_annotation_reader
        annotation_reader.filter_dataset(labels=fxt_default_labels, criterion="XOR")
        project = self.ensure_annotated_project(
            project_service=fxt_project_service, annotation_reader=annotation_reader
        )

        job = attempt_to_train_task(
            training_client=fxt_project_service.training_client,
            task=project.get_trainable_tasks()[0],
            test_mode=fxt_test_mode,
        )

        # Test that requesting a prediction while the project is not ready raises
        # ValueError
        image = fxt_project_service.image_client.get_all_images()[0]
        with pytest.raises(ValueError):
            fxt_project_service.prediction_client.get_image_prediction(image)

        # Monitor train job to make sure the project is train-ready
        timeout = 600 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        interval = 5 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        fxt_project_service.training_client.monitor_jobs(
            [job], timeout=timeout, interval=interval
        )

        # Test that getting model for the train job works
        if fxt_test_mode == SdkTestMode.OFFLINE:
            job.status.state = JobState.FINISHED
        model = fxt_project_service.model_client.get_model_for_job(
            job=job, check_status=False
        )
        assert model is not None

    @pytest.mark.vcr()
    def test_get_model_group_by_algo_name(
        self,
        fxt_project_service: ProjectService,
    ) -> None:
        """
        Test that the model group can be retrieved by algorithm type
        """
        model_client = fxt_project_service.model_client
        project = fxt_project_service.project
        task = project.get_trainable_tasks()[0]
        default_algo = model_client.supported_algos.get_default_for_task_type(
            task_type=task.type
        )

        model_group = fxt_project_service.model_client.get_model_group_by_algo_name(
            algorithm_name=default_algo.algorithm_name
        )

        assert model_group is not None

    @pytest.mark.vcr()
    def test_get_model_algorithm_task_and_version(
        self,
        fxt_project_service: ProjectService,
    ) -> None:
        """
        Test that the specific model can be retrieved by algorithm, task and model
        version
        """
        model_client = fxt_project_service.model_client
        project = fxt_project_service.project
        task = project.get_trainable_tasks()[0]
        algorithm = model_client.supported_algos.get_default_for_task_type(task.type)

        unsupported_algo = get_supported_algorithms(
            fxt_project_service.session, task_type=TaskType.SEGMENTATION
        )[0]

        untrained_algos = copy.deepcopy(
            model_client.supported_algos.get_by_task_type(task.type)
        )
        default_algo = untrained_algos.get_default_for_task_type(task.type)
        untrained_algos.remove(default_algo)
        untrained_algo = untrained_algos[0]

        model_1 = model_client.get_model_by_algorithm_task_and_version(
            algorithm=algorithm, task=task, version=1
        )

        model_no_task = model_client.get_model_by_algorithm_task_and_version(
            algorithm=algorithm, version=1
        )

        latest_model = model_client.get_model_by_algorithm_task_and_version(
            algorithm=algorithm,
        )

        model_not_trained = model_client.get_model_by_algorithm_task_and_version(
            algorithm=untrained_algo, task=task
        )

        model_invalid_version = model_client.get_model_by_algorithm_task_and_version(
            algorithm=algorithm, task=task, version=3
        )

        assert model_1 == model_no_task
        assert latest_model == model_no_task
        assert model_not_trained is None
        assert model_invalid_version is None

        with pytest.raises(ValueError):
            model_client.get_model_by_algorithm_task_and_version(
                algorithm=unsupported_algo, task=task
            )

    def test_download_active_model_for_task(
        self, fxt_project_service: ProjectService, fxt_temp_directory: str
    ) -> None:
        """
        Test that downloading the active model for a task works.
        """
        model_client = fxt_project_service.model_client
        project = fxt_project_service.project
        task = project.get_trainable_tasks()[0]
        algorithm = model_client.supported_algos.get_default_for_task_type(task.type)

        models_folder_name = "models"
        models_filepath = os.path.join(fxt_temp_directory, models_folder_name)

        model_client.download_active_model_for_task(
            path_to_folder=fxt_temp_directory, task=task
        )

        assert os.path.isdir(models_filepath)
        models_content = os.listdir(models_filepath)

        assert len(models_content) >= 3
        assert f"{algorithm.algorithm_name}_base.zip" in models_content
        assert f"{algorithm.algorithm_name} OpenVINO_MO_optimized.zip" in models_content
        assert f"{task.type}_model_details.json" in models_content

    def test_prediction_client_set_mode(
        self, fxt_project_service: ProjectService
    ) -> None:
        """
        Test that changing the prediction mode for the PredictionClient works
        """
        prediction_client = fxt_project_service.prediction_client

        prediction_client.mode = "online"

        assert prediction_client.mode == PredictionMode.ONLINE
