# Copyright (C) 2024 Intel Corporation
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

import pytest
from pytest_mock import MockerFixture

from geti_sdk.benchmarking import Benchmarker
from geti_sdk.data_models.project import Project
from geti_sdk.geti import Geti


@pytest.fixture()
def fxt_benchmarker(
    mocker: MockerFixture,
    fxt_classification_project: Project,
    fxt_mocked_geti: Geti,
) -> Benchmarker:
    _ = mocker.patch(
        "geti_sdk.geti.Geti.get_project",
        return_value=fxt_classification_project,
    )
    _ = mocker.patch("geti_sdk.benchmarking.benchmarker.ModelClient")
    _ = mocker.patch("geti_sdk.benchmarking.benchmarker.TrainingClient")
    algorithms_to_benchmark = ("ALGO_1", "ALGO_2")
    precision_levels = ("PRECISION_1", "PRECISION_2")
    images = ("path_1", "path_2")
    yield Benchmarker(
        geti=fxt_mocked_geti,
        project=mocker.MagicMock(),
        algorithms=algorithms_to_benchmark,
        precision_levels=precision_levels,
        benchmark_images=images,
    )


@pytest.fixture()
def fxt_benchmarker_task_chain(
    mocker: MockerFixture,
    fxt_detection_to_classification_project: Project,
    fxt_mocked_geti: Geti,
) -> Benchmarker:
    _ = mocker.patch(
        "geti_sdk.geti.Geti.get_project",
        return_value=fxt_detection_to_classification_project,
    )
    model_client_object_mock = mocker.MagicMock()
    _ = mocker.patch(
        "geti_sdk.benchmarking.benchmarker.ModelClient",
        return_value=model_client_object_mock,
    )
    active_models = (mocker.MagicMock(), mocker.MagicMock())
    model_client_object_mock.get_all_active_models.return_value = active_models

    _ = mocker.patch("geti_sdk.benchmarking.benchmarker.TrainingClient")
    precision_levels = ("PRECISION_1", "PRECISION_2")
    images = ("path_1", "path_2")

    yield Benchmarker(
        geti=fxt_mocked_geti,
        project=mocker.MagicMock(),
        precision_levels=precision_levels,
        benchmark_images=images,
    )
