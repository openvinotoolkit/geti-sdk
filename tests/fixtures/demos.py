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
from typing import Tuple

import pytest
from vcr import VCR

from geti_sdk import Geti
from geti_sdk.data_models import Project
from geti_sdk.demos import (
    create_anomaly_classification_demo_project,
    create_classification_demo_project,
    create_detection_to_classification_demo_project,
    create_detection_to_segmentation_demo_project,
    create_segmentation_demo_project,
)
from geti_sdk.rest_clients import ProjectClient
from tests.helpers import force_delete_project
from tests.helpers.constants import CASSETTE_EXTENSION, PROJECT_PREFIX


@pytest.fixture(scope="session")
def fxt_demo_images_and_annotations() -> Tuple[int, int]:
    """
    Return the number of images and annotations used in the demo project fixtures

    :return: Tuple containing (n_images, n_annotations)
    """
    yield 4, 4


@pytest.fixture(scope="class")
def fxt_anomaly_classification_demo_project(
    fxt_vcr: VCR,
    fxt_geti: Geti,
    fxt_project_client: ProjectClient,
    fxt_demo_images_and_annotations: Tuple[int, int],
) -> Project:
    """
    Create an annotated anomaly classification project on the Geti instance, and
    return the Project object representing it.
    """
    project_name = f"{PROJECT_PREFIX}_anomaly_classification_demo"
    with fxt_vcr.use_cassette(f"{project_name}.{CASSETTE_EXTENSION}"):
        project = create_anomaly_classification_demo_project(
            geti=fxt_geti,
            n_images=fxt_demo_images_and_annotations[0],
            n_annotations=fxt_demo_images_and_annotations[1],
            project_name=project_name,
        )
    yield project
    with fxt_vcr.use_cassette(f"{project_name}.{CASSETTE_EXTENSION}"):
        force_delete_project(
            project_name=project.name, project_client=fxt_project_client
        )


@pytest.fixture(scope="class")
def fxt_segmentation_demo_project(
    fxt_vcr: VCR,
    fxt_geti: Geti,
    fxt_project_client: ProjectClient,
    fxt_demo_images_and_annotations: Tuple[int, int],
) -> Project:
    """
    Create an annotated segmentation project on the Geti instance, and
    return the Project object representing it.
    """
    project_name = f"{PROJECT_PREFIX}_segmentation_demo"
    with fxt_vcr.use_cassette(f"{project_name}.{CASSETTE_EXTENSION}"):
        project = create_segmentation_demo_project(
            geti=fxt_geti,
            n_images=fxt_demo_images_and_annotations[0],
            n_annotations=fxt_demo_images_and_annotations[1],
            project_name=project_name,
        )
    yield project
    with fxt_vcr.use_cassette(f"{project_name}.{CASSETTE_EXTENSION}"):
        force_delete_project(
            project_name=project.name, project_client=fxt_project_client
        )


@pytest.fixture(scope="class")
def fxt_detection_to_classification_demo_project(
    fxt_vcr: VCR,
    fxt_geti: Geti,
    fxt_project_client: ProjectClient,
    fxt_demo_images_and_annotations: Tuple[int, int],
) -> Project:
    """
    Create an annotated detection_to_classification project on the Geti instance, and
    return the Project object representing it.
    """
    project_name = f"{PROJECT_PREFIX}_detection_to_classification_demo"
    with fxt_vcr.use_cassette(f"{project_name}.{CASSETTE_EXTENSION}"):
        project = create_detection_to_classification_demo_project(
            geti=fxt_geti, n_images=4, n_annotations=4, project_name=project_name
        )
    yield project
    with fxt_vcr.use_cassette(f"{project_name}.{CASSETTE_EXTENSION}"):
        force_delete_project(
            project_name=project.name, project_client=fxt_project_client
        )


@pytest.fixture(scope="class")
def fxt_detection_to_segmentation_demo_project(
    fxt_vcr: VCR,
    fxt_geti: Geti,
    fxt_project_client: ProjectClient,
    fxt_demo_images_and_annotations: Tuple[int, int],
) -> Project:
    """
    Create an annotated detection_to_segmentation project on the Geti instance, and
    return the Project object representing it.
    """
    project_name = f"{PROJECT_PREFIX}_detection_to_segmentation_demo"
    with fxt_vcr.use_cassette(f"{project_name}.{CASSETTE_EXTENSION}"):
        project = create_detection_to_segmentation_demo_project(
            geti=fxt_geti, n_images=4, n_annotations=4, project_name=project_name
        )
    yield project
    with fxt_vcr.use_cassette(f"{project_name}.{CASSETTE_EXTENSION}"):
        force_delete_project(
            project_name=project.name, project_client=fxt_project_client
        )


@pytest.fixture(scope="class")
def fxt_classification_demo_project(
    fxt_vcr: VCR,
    fxt_geti: Geti,
    fxt_project_client: ProjectClient,
    fxt_demo_images_and_annotations: Tuple[int, int],
) -> Project:
    """
    Create an annotated classification project on the Geti instance, and
    return the Project object representing it.
    """
    project_name = f"{PROJECT_PREFIX}_classification_demo"
    with fxt_vcr.use_cassette(f"{project_name}.{CASSETTE_EXTENSION}"):
        project = create_classification_demo_project(
            geti=fxt_geti, n_images=4, n_annotations=4, project_name=project_name
        )
    yield project
    with fxt_vcr.use_cassette(f"{project_name}.{CASSETTE_EXTENSION}"):
        force_delete_project(
            project_name=project.name, project_client=fxt_project_client
        )
