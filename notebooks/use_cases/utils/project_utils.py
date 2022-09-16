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
import time

from geti_sdk import Geti
from geti_sdk.annotation_readers import DirectoryTreeAnnotationReader
from geti_sdk.rest_clients import (
    AnnotationClient,
    ImageClient,
    PredictionClient,
    ProjectClient,
    TrainingClient,
)
from geti_sdk.utils import get_mvtec_dataset


def ensure_trained_anomaly_project(
    geti: Geti, project_name: str = "Transistor anomaly segmentation"
):
    """
    Check whether the project named `project_name` exists on the server, and create it
    if it not.

    If the project does not exist, this method will create an anomaly classification
    project based on the MVTec AD `transistor` dataset.

    :param geti: Geti instance pointing to the Intel® Geti™ server
    :param project_name: Name of the project to look for or create
    :return: Project object representing the project on the server
    """
    project_client = ProjectClient(session=geti.session, workspace_id=geti.workspace_id)
    project = project_client.get_project_by_name(project_name)
    if project is None:
        logging.info(f"Project '{project_name}' does not exist yet, creating it...")

        # First get the mvtec transistor dataset. Will be downloaded to a default
        # datapath if it is not found on the system yet
        mvpath = get_mvtec_dataset()

        # Prepare the annotation reader
        annotation_reader = DirectoryTreeAnnotationReader(
            base_data_folder=mvpath, subset_folder_names=["train", "test"]
        )
        annotation_reader.group_labels(
            labels_to_group=["damaged_case", "cut_lead", "misplaced", "bent_lead"],
            group_name="Anomalous",
        )
        annotation_reader.group_labels(labels_to_group=["good"], group_name="Normal")

        # Create the project
        project = project_client.create_project(
            project_name=project_name,
            project_type="anomaly_classification",
            labels=[[]],
        )

        # Upload images to the project
        data_filepaths = annotation_reader.get_data_filenames()
        image_client = ImageClient(
            session=geti.session, workspace_id=geti.workspace_id, project=project
        )
        images = image_client.upload_from_list(
            path_to_folder=mvpath,
            image_names=data_filepaths,
            extension_included=False,
            image_names_as_full_paths=True,
        )
        annotation_client = AnnotationClient(
            session=geti.session,
            workspace_id=geti.workspace_id,
            project=project,
            annotation_reader=annotation_reader,
        )

        # The annotation_reader is used to upload annotations for all images
        for image, full_path in zip(images, data_filepaths):
            image.name = full_path
        annotation_client.upload_annotations_for_images(images=images)

    prediction_client = PredictionClient(
        session=geti.session, workspace_id=geti.workspace_id, project=project
    )
    if not prediction_client.ready_to_predict:
        training_client = TrainingClient(
            session=geti.session, workspace_id=geti.workspace_id, project=project
        )
        train_job = training_client.train_task(0)
        training_client.monitor_jobs(jobs=[train_job])

    tries = 20
    while not prediction_client.ready_to_predict and tries > 0:
        time.sleep(1)
        tries -= 1

    if prediction_client.ready_to_predict:
        print(f"\nProject '{project.name}' is trained and ready to predict.\n")
    else:
        print(
            f"\nAll jobs completed, yet project '{project.name}' is still not ready "
            f"to predict. This is likely due to an error in the training job.\n"
        )
    return project
