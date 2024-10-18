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
from typing import List, Optional, Sequence

from geti_sdk import Geti
from geti_sdk.annotation_readers import AnnotationReader
from geti_sdk.rest_clients import ProjectClient

from .constants import PROJECT_PREFIX
from .finalizers import force_delete_project
from .project_service import ProjectService


def get_or_create_annotated_project_for_test_class(
    project_service: ProjectService,
    annotation_readers: Sequence[AnnotationReader],
    project_name: str,
    project_type: str = "detection",
    enable_auto_train: bool = False,
    learning_parameter_settings: str = "minimal",
    annotation_requirements_first_training: Optional[int] = None,
):
    """
    This function returns an annotated project with `project_name` of type
    `project_type`.

    :param project_service: ProjectService instance to which the project should be added
    :param annotation_readers: List of AnnotationReader instances from which to get the
        annotations. The number of annotation readers must match the number of
        trainable tasks in the project.
    :param project_name: Name of the project
    :param project_type: Type of the project
    :param enable_auto_train: True to turn auto-training on, False to leave it off
    :param learning_parameter_settings: Settings to use for the learning parameters
        during model training. There are three options:
          'minimal'     - Set hyper parameters such that the training time is minimized
                          (i.e. single epoch, low batch size, etc.)
          'default'     - Use default hyper parameter settings
          'reduced_mem' - Reduce the batch size for memory intensive tasks
    :return: Project instance corresponding to the project on the Intel® Geti™ server
    """
    project_exists = project_service.has_project
    labels = [reader.get_all_label_names() for reader in annotation_readers]

    project = project_service.get_or_create_project(
        project_name=project_name, project_type=project_type, labels=labels
    )
    if not project_exists:
        project_service.set_auto_train(False)
        if learning_parameter_settings == "minimal":
            project_service.set_minimal_training_hypers()
        elif learning_parameter_settings == "reduced_mem":
            project_service.set_reduced_memory_hypers()
        elif learning_parameter_settings != "default":
            logging.info(
                f"Invalid learning parameter settings '{learning_parameter_settings}' "
                f"specified, continuing with default hyper parameters."
            )
        if annotation_requirements_first_training is not None:
            project_service.set_auto_training_annotation_requirement(
                required_images=annotation_requirements_first_training
            )

        project_service.add_annotated_media(
            annotation_readers=annotation_readers, n_images=-1
        )
        project_service.set_auto_train(enable_auto_train)
    return project


def remove_all_test_projects(geti: Geti) -> List[str]:
    """
    Removes all projects created in the SDK tests from the server.

    WARNING: This will remove projects without asking for confirmation. Use with
    caution!

    :param geti: Geti instance pointing to the server from which to remove all
        projects created by the SDK test suite
    """
    project_client = ProjectClient(session=geti.session, workspace_id=geti.workspace_id)
    projects_removed: List[str] = []
    for project in project_client.get_all_projects(get_project_details=False):
        if project.name.startswith(PROJECT_PREFIX):
            force_delete_project(project, project_client)
            projects_removed.append(project.name)
    logging.info(f"{len(projects_removed)} test projects were removed from the server.")
    return projects_removed
