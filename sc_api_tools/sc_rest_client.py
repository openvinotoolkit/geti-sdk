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
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .annotation_readers import (
    AnnotationReader,
    DatumAnnotationReader,
    SCAnnotationReader,
)
from .data_models import Image, Prediction, Project, TaskType, Video, VideoFrame
from .data_models.containers import MediaList
from .data_models.enums import OptimizationType
from .deployment import DeployedModel, Deployment
from .http_session import (
    SCRequestException,
    SCSession,
    ServerCredentialConfig,
    ServerTokenConfig,
)
from .rest_clients import (
    AnnotationClient,
    ConfigurationClient,
    ImageClient,
    ModelClient,
    PredictionClient,
    ProjectClient,
    VideoClient,
)
from .utils import (
    generate_classification_labels,
    get_default_workspace_id,
    get_task_types_by_project_type,
    show_image_with_annotation_scene,
    show_video_frames_with_annotation_scenes,
)


class SCRESTClient:
    """
    Interact with a Sonoma Creek cluster via the REST API.

    The SCRESTClient class provides methods for project creation, downloading and
    uploading, as well as project deployment. Initializing the class will establish a
    HTTP session to the SC cluster, and requires authentication.

    NOTE: The client can either be initialized using user credentials (`username`
    and `password`), or using a personal access token (`token`). Arguments for either
    one of these two options must be passed, otherwise a TypeError will be raised.

    :param host: IP address or URL at which the cluster can be reached, for example
        'https://0.0.0.0' or 'https://sc_example.intel.com'
    :param username: Username to log in to the cluster
    :param password: Password to log in to the cluster
    :param token: Personal access token that can be used for authentication
    :param workspace_id: Optional ID of the workspace that should be addressed by this
        SCRESTClient instance. If not specified, the default workspace is used.
    :param verify_certificate: True to verify the certificate used for making HTTPS
        requests encrypted using TLS protocol. If set to False, an
        InsecureRequestWarning will be issued
    :param proxies: Optional dictionary containing proxy information. For example
        {
            'http': http://proxy-server.com:<http_port_number>,
            'https': http://proxy-server.com:<https_port_number>
        },
        if set to None (the default), no proxy settings will be used.
    """

    def __init__(
        self,
        host: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        workspace_id: Optional[str] = None,
        verify_certificate: bool = False,
        proxies: Optional[Dict[str, str]] = None,
    ):
        if token is not None:
            server_config = ServerTokenConfig(
                host=host,
                token=token,
                proxies=proxies,
                has_valid_certificate=verify_certificate,
            )
            if username is not None or password is not None:
                print(
                    "Both a personal access token and credentials were passed to the "
                    "SCRESTClient, using token authentication."
                )
        elif username is not None and password is not None:
            server_config = ServerCredentialConfig(
                host=host,
                username=username,
                password=password,
                proxies=proxies,
                has_valid_certificate=verify_certificate,
            )
        else:
            raise TypeError(
                "__init__ missing required keyword arguments: Either `username` and "
                "`password` or `token` must be specified."
            )
        self.session = SCSession(
            server_config=server_config,
        )
        if workspace_id is None:
            workspace_id = get_default_workspace_id(self.session)
        self.workspace_id = workspace_id

    def logout(self) -> None:
        """
        Log out of SonomaCreek and end the HTTP session.
        """
        self.session.logout()

    def download_project(
        self,
        project_name: str,
        target_folder: Optional[str] = None,
        include_predictions: bool = False,
        include_active_models: bool = False,
        include_deployment: bool = False,
    ) -> Project:
        """
        Download a project with name `project_name` to the local disk. All images,
        image annotations, videos and video frame annotations in the project are
        downloaded. By default, predictions and models are not downloaded, but they
        can be included by passing `include_predictions=True` and
        `include_active_models=True`, respectively.

        In addition, if `include_deployment` is set to True, this method will create
        and download a deployment for the project as well.

        This method will download data to the path `target_folder`, the contents of the
        folder will be:

            images
                Folder holding all images in the project, if any

            videos
                Folder holding all videos in the project, if any

            annotations
                Directory holding all annotations in the project, in .json format

            predictions
                Directory holding all predictions in the project, in .json format. If
                available, this will include saliency maps in .jpeg format. Only
                created if `include_predictions=True`

            models
                Folder containing the active model for the project. This folder contains
                zip files holding the data for the active models for the tasks in the
                project, and any optimized models derived from them. Models are only
                downloaded if `include_active_models = True`.

            deployment
                Folder containing the deployment for the project, that can be used for
                local inference. The deployment is only created if
                `include_deployment = True`.

            project.json
                File containing the project parameters, that can be used to re-create
                the project.

            configuration.json
                File containing the configurable parameters for the active models in the
                project

        Downloading a project may take a substantial amount of time if the project
        dataset is large.

        :param project_name: Name of the project to download
        :param target_folder: Path to the local folder in which the project data
            should be saved. If not specified, a new directory named `project_name`
            will be created inside the current working directory.
        :param include_predictions: True to also download the predictions for all
            images and videos in the project, False to not download any predictions.
            If this is set to True but the project has no trained models, downloading
            predictions will be skipped.
        :param include_active_models: True to download the active models for all
            tasks in the project, and any optimized models derived from them. False to
            not download any models. Defaults to False
        :param include_deployment: True to create and download a deployment for the
            project, that can be used for local inference with OpenVINO. Defaults to
            False.
        :return: Project object, holding information obtained from the cluster
            regarding the downloaded project
        """
        # Obtain project details from cluster
        project_client = ProjectClient(
            session=self.session, workspace_id=self.workspace_id
        )
        project = project_client.get_project_by_name(project_name)

        # Validate or create target_folder
        if target_folder is None:
            target_folder = os.path.join(".", project_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Download project creation parameters:
        project_client.download_project_info(
            project_name=project_name, path_to_folder=target_folder
        )

        # Download images
        image_client = ImageClient(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        images = image_client.get_all_images()
        if len(images) > 0:
            image_client.download_all(
                path_to_folder=target_folder,
                append_image_uid=images.has_duplicate_filenames,
            )

        # Download videos
        video_client = VideoClient(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        videos = video_client.get_all_videos()
        if len(videos) > 0:
            video_client.download_all(
                path_to_folder=target_folder,
                append_video_uid=videos.has_duplicate_filenames,
            )

        # Download annotations
        annotation_client = AnnotationClient(
            session=self.session, project=project, workspace_id=self.workspace_id
        )
        if len(images) > 0:
            annotation_client.download_annotations_for_images(
                images=images,
                path_to_folder=target_folder,
                append_image_uid=images.has_duplicate_filenames,
            )
        if len(videos) > 0:
            annotation_client.download_annotations_for_videos(
                videos=videos,
                path_to_folder=target_folder,
                append_video_uid=videos.has_duplicate_filenames,
            )

        # Download predictions
        prediction_client = PredictionClient(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        if prediction_client.ready_to_predict and include_predictions:
            if len(images) > 0:
                prediction_client.download_predictions_for_images(
                    images=images,
                    path_to_folder=target_folder,
                    include_result_media=True,
                )
            if len(videos) > 0:
                prediction_client.download_predictions_for_videos(
                    videos=videos,
                    path_to_folder=target_folder,
                    include_result_media=True,
                    inferred_frames_only=False,
                )

        # Download configuration
        configuration_client = ConfigurationClient(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        configuration_client.download_configuration(path_to_folder=target_folder)

        # Download active models
        if include_active_models:
            model_client = ModelClient(
                workspace_id=self.workspace_id, session=self.session, project=project
            )
            model_client.download_all_active_models(path_to_folder=target_folder)

        # Download deployment
        if include_deployment:
            print("Creating deployment for project...")
            self.deploy_project(project.name, output_folder=target_folder)

        print(f"Project '{project.name}' was downloaded successfully.")
        return project

    def upload_project(
        self,
        target_folder: str,
        project_name: Optional[str] = None,
        enable_auto_train: bool = True,
    ) -> Project:
        """
        Upload a previously downloaded SC project to the cluster. This method expects
        the `target_folder` to contain the following:

            images
                Folder holding all images in the project, if any

            videos
                Folder holding all videos in the project, if any

            annotations
                Directory holding all annotations in the project, in .json format

            project.json
                File containing the project parameters, that can be used to re-create
                the project.

            configuration.json
                Optional file containing the configurable parameters for the active
                models in the project. If this file is not present, the configurable
                parameters for the project will be left at their default values.

        :param target_folder: Folder holding the project data to upload
        :param project_name: Optional name of the project to create on the cluster. If
            left unspecified, the name of the project found in the configuration in
            the `target_folder` will be used.
        :param enable_auto_train: True to enable auto-training for all tasks directly
            after all annotations have been uploaded. This will directly trigger a
            training round if the conditions for auto-training are met. False to leave
            auto-training disabled for all tasks. Defaults to True.
        :return: Project object, holding information obtained from the cluster
            regarding the uploaded project
        """
        project_client = ProjectClient(
            session=self.session, workspace_id=self.workspace_id
        )
        project = project_client.create_project_from_folder(
            path_to_folder=target_folder, project_name=project_name
        )

        # Disable auto-train to prevent the project from training right away
        configuration_client = ConfigurationClient(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        configuration_client.set_project_auto_train(auto_train=False)

        # Upload images
        image_client = ImageClient(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        images = image_client.upload_folder(
            path_to_folder=os.path.join(target_folder, "images")
        )

        # Upload videos
        video_client = VideoClient(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        videos = video_client.upload_folder(
            path_to_folder=os.path.join(target_folder, "videos")
        )

        media_lists: List[Union[MediaList[Image], MediaList[Video]]] = []
        if len(images) > 0:
            media_lists.append(images)
        if len(videos) > 0:
            media_lists.append(videos)

        # Upload annotations
        annotation_reader = SCAnnotationReader(
            base_data_folder=os.path.join(target_folder, "annotations"),
            task_type=None,
        )
        annotation_client = AnnotationClient[SCAnnotationReader](
            session=self.session,
            project=project,
            workspace_id=self.workspace_id,
            annotation_reader=annotation_reader,
        )
        if len(images) > 0:
            annotation_client.upload_annotations_for_images(
                images=images,
            )
        if len(videos) > 0:
            annotation_client.upload_annotations_for_videos(
                videos=videos,
            )

        configuration_file = os.path.join(target_folder, "configuration.json")
        if os.path.isfile(configuration_file):
            result = None
            try:
                result = configuration_client.apply_from_file(
                    path_to_folder=target_folder
                )
            except SCRequestException:
                print(
                    f"Attempted to set configuration according to the "
                    f"'configuration.json' file in the project directory, but setting "
                    f"the configuration failed. Probably the configuration specified "
                    f"in '{configuration_file}' does "
                    f"not apply to the default model for one of the tasks in the "
                    f"project. Please make sure to reconfigure the models manually."
                )
            if result is None:
                warnings.warn(
                    f"Not all configurable parameters could be set according to the "
                    f"configuration in {configuration_file}. Please make sure to "
                    f"verify model configuration manually."
                )
        configuration_client.set_project_auto_train(auto_train=enable_auto_train)
        print(f"Project '{project.name}' was uploaded successfully.")
        return project

    def create_single_task_project_from_dataset(
        self,
        project_name: str,
        project_type: str,
        path_to_images: str,
        annotation_reader: AnnotationReader,
        labels: Optional[List[str]] = None,
        number_of_images_to_upload: int = -1,
        number_of_images_to_annotate: int = -1,
        enable_auto_train: bool = True,
    ) -> Project:
        """
        Create a single task project named `project_name` on the SC cluster, and
        upload data from a dataset on local disk.

        The type of task that will be in the project can be controlled by setting the
        `project_type`, options are:

            * classification
            * detection
            * segmentation
            * anomaly_classification
            * anomaly_detection
            * anomaly_segmentation
            * instance_segmentation
            * rotated_detection

        If a project called `project_name` exists on the server, this method will
        attempt to upload the media and annotations to the existing project.

        :param project_name: Name of the project to create
        :param project_type: Type of the project, this determines which task the
            project will perform. See above for possible values
        :param path_to_images: Path to the folder holding the images on the local disk.
            See above for details.
        :param annotation_reader: AnnotationReader instance that will be used to
            obtain annotations for the images.
        :param labels: Optional list of labels to use. This will only be used if the
            `annotation_reader` that is passed also supports dataset filtering. If
            not specified, all labels that are found in the dataset are used.
        :param number_of_images_to_upload: Optional integer specifying how many images
            should be uploaded. If not specified, all images found in the dataset are
            uploaded.
        :param number_of_images_to_annotate: Optional integer specifying how many
            images should be annotated. If not specified, annotations for all images
            that have annotations available will be uploaded.
        :param enable_auto_train: True to enable auto-training for all tasks directly
            after all annotations have been uploaded. This will directly trigger a
            training round if the conditions for auto-training are met. False to leave
            auto-training disabled for all tasks. Defaults to True.
        :return: Project object, holding information obtained from the cluster
            regarding the uploaded project
        """
        if labels is None:
            labels = annotation_reader.get_all_label_names()
        else:
            if project_type == "classification":
                # Handle label generation for classification case
                filter_settings = annotation_reader.applied_filters
                criterion = filter_settings[0]["criterion"]
                multilabel = True
                if criterion == "XOR":
                    multilabel = False
                labels = generate_classification_labels(labels, multilabel=multilabel)
            elif project_type == "anomaly_classification":
                labels = ["Normal", "Anomalous"]

        # Create project
        project_client = ProjectClient(
            session=self.session, workspace_id=self.workspace_id
        )
        project = project_client.create_project(
            project_name=project_name, project_type=project_type, labels=[labels]
        )
        # Disable auto training
        configuration_client = ConfigurationClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        configuration_client.set_project_auto_train(auto_train=False)

        # Upload images
        image_client = ImageClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        if isinstance(annotation_reader, DatumAnnotationReader):
            images = image_client.upload_from_list(
                path_to_folder=path_to_images,
                image_names=annotation_reader.get_all_image_names(),
                n_images=number_of_images_to_upload,
            )
        else:
            images = image_client.upload_folder(
                path_to_images, n_images=number_of_images_to_upload
            )

        if (
            number_of_images_to_annotate < len(images)
            and number_of_images_to_annotate != -1
        ):
            images = images[:number_of_images_to_annotate]

        # Set annotation reader task type
        annotation_reader.task_type = project.get_trainable_tasks()[0].type
        annotation_reader.prepare_and_set_dataset(
            task_type=project.get_trainable_tasks()[0].type
        )
        # Upload annotations
        annotation_client = AnnotationClient(
            session=self.session,
            project=project,
            workspace_id=self.workspace_id,
            annotation_reader=annotation_reader,
        )
        annotation_client.upload_annotations_for_images(images)

        configuration_client.set_project_auto_train(auto_train=enable_auto_train)
        return project

    def create_task_chain_project_from_dataset(
        self,
        project_name: str,
        project_type: str,
        path_to_images: str,
        label_source_per_task: List[Union[AnnotationReader, List[str]]],
        number_of_images_to_upload: int = -1,
        number_of_images_to_annotate: int = -1,
        enable_auto_train: bool = True,
    ) -> Project:
        """
        Create a single task project named `project_name` on the SC cluster, and
        upload data from a dataset on local disk.

        The type of task that will be in the project can be controlled by setting the
        `project_type`, current options are:

            * detection_to_segmentation
            * detection_to_classification

        If a project called `project_name` exists on the server, this method will
        attempt to upload the media and annotations to the existing project.

        :param project_name: Name of the project to create
        :param project_type: Type of the project, this determines which task the
            project will perform. See above for possible values
        :param path_to_images: Path to the folder holding the images on the local disk.
            See above for details.
        :param label_source_per_task: List containing the label sources for each task
            in the task chain. Each entry in the list corresponds to the label source
            for one task. The list can contain either AnnotationReader instances that
            will be used to obtain the labels for a task, or it can contain a list of
            labels to use for that task.

            For example, in a detection -> classification project we may have labels
            for the first task (for instance 'dog'), but no annotations for the second
            task yet (e.g. ['small', 'large']). In that case the
            `label_source_per_task` should contain:

                [AnnotationReader(), ['small', 'large']]

            Where the annotation reader has been properly instantiated to read the
            annotations for the 'dog' labels.

        :param number_of_images_to_upload: Optional integer specifying how many images
            should be uploaded. If not specified, all images found in the dataset are
            uploaded.
        :param number_of_images_to_annotate: Optional integer specifying how many
            images should be annotated. If not specified, annotations for all images
            that have annotations available will be uploaded.
        :param enable_auto_train: True to enable auto-training for all tasks directly
            after all annotations have been uploaded. This will directly trigger a
            training round if the conditions for auto-training are met. False to leave
            auto-training disabled for all tasks. Defaults to True.
        :return: Project object, holding information obtained from the cluster
            regarding the uploaded project
        """
        labels_per_task = [
            entry.get_all_label_names()
            if isinstance(entry, AnnotationReader)
            else entry
            for entry in label_source_per_task
        ]
        annotation_readers_per_task = [
            entry if isinstance(entry, AnnotationReader) else None
            for entry in label_source_per_task
        ]

        task_types = get_task_types_by_project_type(project_type)
        labels_per_task = self._check_unique_label_names(
            labels_per_task=labels_per_task,
            task_types=task_types,
            annotation_readers_per_task=annotation_readers_per_task,
        )

        # Create project
        project_client = ProjectClient(
            session=self.session, workspace_id=self.workspace_id
        )
        project = project_client.create_project(
            project_name=project_name, project_type=project_type, labels=labels_per_task
        )
        # Disable auto training
        configuration_client = ConfigurationClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        configuration_client.set_project_auto_train(auto_train=False)

        # Upload images
        image_client = ImageClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        # Assume that the first task determines the media that will be uploaded
        first_task_reader = annotation_readers_per_task[0]
        if isinstance(first_task_reader, DatumAnnotationReader):
            images = image_client.upload_from_list(
                path_to_folder=path_to_images,
                image_names=first_task_reader.get_all_image_names(),
                n_images=number_of_images_to_upload,
            )
        else:
            images = image_client.upload_folder(
                path_to_images, n_images=number_of_images_to_upload
            )

        if (
            number_of_images_to_annotate < len(images)
            and number_of_images_to_annotate != -1
        ):
            images = images[:number_of_images_to_annotate]

        append_annotations = False
        previous_task_type = None
        for task_type, reader in zip(task_types, annotation_readers_per_task):
            if reader is not None:
                # Set annotation reader task type
                reader.task_type = task_type
                reader.prepare_and_set_dataset(
                    task_type=task_type, previous_task_type=previous_task_type
                )
                # Upload annotations
                annotation_client = AnnotationClient(
                    session=self.session,
                    project=project,
                    workspace_id=self.workspace_id,
                    annotation_reader=reader,
                )
                annotation_client.upload_annotations_for_images(
                    images=images, append_annotations=append_annotations
                )
                append_annotations = True
            previous_task_type = task_type
        configuration_client.set_project_auto_train(auto_train=enable_auto_train)
        return project

    @staticmethod
    def _check_unique_label_names(
        labels_per_task: List[List[str]],
        task_types: List[TaskType],
        annotation_readers_per_task: List[AnnotationReader],
    ):
        """
        Check that the names of all labels passed in `labels_per_task` are unique. If
        they are not unique and there is a segmentation task in the task chain, this
        method tries to generate segmentation labels in order to guarantee unique label
        names

        :param labels_per_task: Nested list of label names per task
        :param task_types: List of TaskTypes for every trainable task in the project
        :param annotation_readers_per_task: List of annotation readers for all
            trainable tasks in the project
        :raises ValueError: If the label names are not unique and this method is not
            able to generate unique label names for this configuration
        :return: List of labels per task with unique label names
        """
        # Check that label names are unique, try to generate segmentation labels if not
        all_labels = [label for labels in labels_per_task for label in labels]
        if len(set(all_labels)) != len(all_labels):
            new_labels = []
            new_labels_per_task = []
            for index, task_type in enumerate(task_types):
                reader = annotation_readers_per_task[index]
                if task_type == TaskType.SEGMENTATION:
                    if isinstance(reader, DatumAnnotationReader):
                        reader.convert_labels_to_segmentation_names()
                new_labels.extend(reader.get_all_label_names())
                new_labels_per_task.append(reader.get_all_label_names())
            if len(set(new_labels)) != len(new_labels):
                raise ValueError(
                    "Unable to create project. Label names must be unique!"
                )
            else:
                return new_labels_per_task
        else:
            return labels_per_task

    def download_all_projects(
        self, target_folder: str, include_predictions: bool = True
    ) -> List[Project]:
        """
        Download all projects in the workspace from the SC cluster.

        :param target_folder: Directory on local disk to download the project data to.
            If not specified, this method will create a directory named 'projects' in
            the current working directory.
        :param include_predictions: True to also download the predictions for all
            images and videos in the project, False to not download any predictions.
            If this is set to True but the project has no trained models, downloading
            predictions will be skipped.
        :return: List of Project objects, each entry corresponding to one of the
            projects found on the SC cluster
        """
        # Obtain project details from cluster
        project_client = ProjectClient(
            session=self.session, workspace_id=self.workspace_id
        )
        projects = project_client.get_all_projects()

        # Validate or create target_folder
        if target_folder is None:
            target_folder = os.path.join(".", "projects")
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        print(
            f"Found {len(projects)} projects on the SC cluster. Commencing project "
            f"download..."
        )

        # Download all found projects
        for index, project in enumerate(projects):
            print(f"Downloading project '{project.name}'... {index+1}/{len(projects)}.")
            self.download_project(
                project_name=project.name,
                target_folder=os.path.join(target_folder, project.name),
                include_predictions=include_predictions,
            )
        return projects

    def upload_all_projects(self, target_folder: str) -> List[Project]:
        """
        Upload all projects found in the directory `target_folder` on local disk to
        the SC cluster.

        This method expects the directory `target_folder` to contain subfolders. Each
        subfolder should correspond to the (previously downloaded) data for one
        project. The method looks for project folders non-recursively, meaning that
        only folders directly below the `target_folder` in the hierarchy are
        considered to be uploaded as project.

        :param target_folder: Directory on local disk to retrieve the project data from
        :return: List of Project objects, each entry corresponding to one of the
            projects uploaded to the SC cluster
        """
        candidate_project_folders = [
            os.path.join(target_folder, subfolder)
            for subfolder in os.listdir(target_folder)
        ]
        project_folders = [
            folder
            for folder in candidate_project_folders
            if ProjectClient.is_project_dir(folder)
        ]
        print(
            f"Found {len(project_folders)} project data folders in the target "
            f"directory '{target_folder}'. Commencing project upload..."
        )
        projects: List[Project] = []
        for index, project_folder in enumerate(project_folders):
            print(
                f"Uploading project from folder '{os.path.basename(project_folder)}'..."
                f" {index + 1}/{len(project_folders)}."
            )
            project = self.upload_project(
                target_folder=project_folder, enable_auto_train=False
            )
            projects.append(project)
        return projects

    def upload_and_predict_media_folder(
        self,
        project_name: str,
        media_folder: str,
        output_folder: Optional[str] = None,
        delete_after_prediction: bool = False,
        skip_if_filename_exists: bool = False,
    ) -> bool:
        """
        Upload a folder with media (images, videos or both) from local disk at path
        `target_folder` to the project with name `project_name` on the SC cluster.
        After the media upload is complete, predictions will be downloaded for all
        media in the folder. This method will create a 'predictions' directory in
        the `target_folder`, containing the prediction output in json format.

        If `delete_after_prediction` is set to True, all uploaded media will be
        removed from the project on the SC cluster after the predictions have
        been downloaded.

        :param project_name: Name of the project to upload media to
        :param media_folder: Path to the folder to upload media from
        :param output_folder: Path to save the predictions to. If not specified, this
            method will create a folder named '<media_folder_name>_predictions' on
            the same level as the media_folder
        :param delete_after_prediction: True to remove the media from the project
            once all predictions are received, False to keep the media in the project.
        :param skip_if_filename_exists: Set to True to skip uploading of an image (or
            video) if an image (or video) with the same filename already exists in the
            project. Defaults to False
        :return: True if all media was uploaded, and predictions for all media were
            successfully downloaded. False otherwise
        """
        # Obtain project details from cluster
        project_client = ProjectClient(
            session=self.session, workspace_id=self.workspace_id
        )
        project = project_client.get_project_by_name(project_name=project_name)
        if project is None:
            print(
                f"Project '{project_name}' was not found on the cluster. Aborting "
                f"media upload."
            )
            return False

        # Upload images
        image_client = ImageClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        images = image_client.upload_folder(
            path_to_folder=media_folder, skip_if_filename_exists=skip_if_filename_exists
        )

        # Upload videos
        video_client = VideoClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        videos = video_client.upload_folder(
            path_to_folder=media_folder, skip_if_filename_exists=skip_if_filename_exists
        )

        prediction_client = PredictionClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        if not prediction_client.ready_to_predict:
            print(
                f"Project '{project_name}' is not ready to make predictions, likely "
                f"because one of the tasks in the task chain does not have a "
                f"trained model yet. Aborting prediction."
            )

        # Set and create output folder if necessary
        if output_folder is None:
            output_folder = media_folder + "_predictions"
        if not os.path.exists(output_folder) and os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # Request image predictions
        if len(images) > 0:
            prediction_client.download_predictions_for_images(
                images=images, path_to_folder=output_folder
            )

        # Request video predictions
        if len(videos) > 0:
            prediction_client.download_predictions_for_videos(
                videos=videos, path_to_folder=output_folder, inferred_frames_only=False
            )

        # Delete media if required
        result = True
        if delete_after_prediction:
            images_deleted = True
            videos_deleted = True
            if len(images) > 0:
                images_deleted = image_client.delete_images(images=images)
            if len(videos) > 0:
                videos_deleted = video_client.delete_videos(videos=videos)
            result = images_deleted and videos_deleted
        return result

    def upload_and_predict_image(
        self,
        project_name: str,
        image: Union[np.ndarray, Image, VideoFrame, str, os.PathLike],
        visualise_output: bool = True,
        delete_after_prediction: bool = False,
    ) -> Tuple[Image, Prediction]:
        """
        Upload a single image to a project named `project_name` on the SC cluster,
        and return a prediction for it.

        :param project_name: Name of the project to upload the image to
        :param image: Image, numpy array representing an image, or filepath to an
            image to upload and get a prediction for
        :param visualise_output: True to show the resulting prediction, overlayed on
            the image
        :param delete_after_prediction: True to remove the image from the project
            once the prediction is received, False to keep the image in the project.
        :return: Tuple containing:

            - Image object representing the image that was uploaded
            - Prediction for the image
        """
        project_client = ProjectClient(self.session, workspace_id=self.workspace_id)
        project = project_client.get_project_by_name(project_name)
        if project is None:
            raise ValueError(
                f"Project '{project_name}' was not found on the cluster. Aborting "
                f"image upload."
            )

        # Upload the image
        image_client = ImageClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        needs_upload = True
        if isinstance(image, Image):
            if image.id in image_client.get_all_images().ids:
                # Image is already in the project, make sure not to delete it
                needs_upload = False
                image_data = None
            else:
                image_data = image.get_data(self.session)
        else:
            image_data = image
        if needs_upload:
            if image_data is None:
                raise ValueError(
                    f"Cannot upload entity {image}. No data available for upload."
                )
            uploaded_image = image_client.upload_image(image=image_data)
        else:
            uploaded_image = image

        # Get prediction
        prediction_client = PredictionClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        if not prediction_client.ready_to_predict:
            raise ValueError(
                f"Project '{project_name}' is not ready to make predictions. At least "
                f"one of the tasks in the task chain does not have any models trained."
            )
        prediction = prediction_client.get_image_prediction(uploaded_image)
        uploaded_image.get_data(self.session)

        if delete_after_prediction and needs_upload:
            image_client.delete_images(images=MediaList([uploaded_image]))

        if visualise_output:
            show_image_with_annotation_scene(
                image=uploaded_image, annotation_scene=prediction
            )

        return uploaded_image, prediction

    def upload_and_predict_video(
        self,
        project_name: str,
        video: Union[Video, str, os.PathLike, Union[Sequence[np.ndarray], np.ndarray]],
        frame_stride: Optional[int] = None,
        visualise_output: bool = True,
        delete_after_prediction: bool = False,
    ) -> Tuple[Video, MediaList[VideoFrame], List[Prediction]]:
        """
        Upload a single video to a project named `project_name` on the SC cluster,
        and return a list of predictions for the frames in the video.

        The parameter 'frame_stride' is used to control the stride for frame
        extraction. Predictions are only generated for the extracted frames. So to
        get predictions for all frames, `frame_stride=1` can be passed.

        :param project_name: Name of the project to upload the image to
        :param video: Video or filepath to a video to upload and get predictions for.
            Can also be a 4D numpy array or a list of 3D numpy arrays, shaped such
            that the array dimensions represent `frames x width x height x channels`,
            i.e. each entry holds the pixel data for a video frame.
        :param frame_stride: Frame stride to use. This determines the number of
            frames that will be extracted from the video, and for which predictions
            will be generated
        :param visualise_output: True to show the resulting prediction, overlayed on
            the video frames.
        :param delete_after_prediction: True to remove the video from the project
            once the prediction is received, False to keep the video in the project.
        :return: Tuple containing:

            - Video object holding the data for the uploaded video
            - List of VideoFrames extracted from the video, for which predictions
              have been generated
            - List of Predictions for the Video
        """
        project_client = ProjectClient(self.session, workspace_id=self.workspace_id)
        project = project_client.get_project_by_name(project_name)
        if project is None:
            raise ValueError(
                f"Project '{project_name}' was not found on the cluster. Aborting "
                f"image upload."
            )
        # Upload the video
        video_client = VideoClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        needs_upload = True
        if isinstance(video, Video):
            if video.id in video_client.get_all_videos().ids:
                # Video is already in the project, make sure not to delete it
                needs_upload = False
                video_data = None
            else:
                video_data = video.get_data(self.session)
        elif isinstance(video, (Sequence, np.ndarray)):
            if not isinstance(video, np.ndarray):
                video_data = np.array(video)
            else:
                video_data = video
        else:
            video_data = video
        if needs_upload:
            print(f"Uploading video to project '{project_name}'...")
            uploaded_video = video_client.upload_video(video=video_data)
        else:
            uploaded_video = video

        # Get prediction for frames
        prediction_client = PredictionClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        if not prediction_client.ready_to_predict:
            raise ValueError(
                f"Project '{project_name}' is not ready to make predictions. At least "
                f"one of the tasks in the task chain does not have any models trained."
            )
        if frame_stride is None:
            frame_stride = uploaded_video.media_information.frame_stride
        frames = MediaList(
            uploaded_video.to_frames(frame_stride=frame_stride, include_data=True)
        )
        print(
            f"Getting predictions for video '{uploaded_video.name}', using stride "
            f"{frame_stride}"
        )
        predictions = [
            prediction_client.get_video_frame_prediction(frame) for frame in frames
        ]
        if delete_after_prediction and needs_upload:
            video_client.delete_videos(videos=MediaList([uploaded_video]))
        if visualise_output:
            show_video_frames_with_annotation_scenes(
                video_frames=frames, annotation_scenes=predictions
            )
        return uploaded_video, frames, predictions

    def deploy_project(
        self, project_name: str, output_folder: Optional[Union[str, os.PathLike]] = None
    ) -> Deployment:
        """
        Deploy a project by creating a Deployment instance. The Deployment contains
        the optimized active models for each task in the project, and can be loaded
        with OpenVINO to run inference locally.

        :param project_name: Name of the project to deploy
        :param output_folder: Path to a folder on local disk to which the Deployment
            should be downloaded. If no path is specified, the deployment will not be
            saved.
        :return: Deployment for the project
        """
        project_client = ProjectClient(self.session, workspace_id=self.workspace_id)
        project = project_client.get_project_by_name(project_name)
        if project is None:
            raise ValueError(
                f"Project '{project_name}' was not found on the cluster. Aborting "
                f"project deployment."
            )
        model_client = ModelClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        active_models = [
            model for model in model_client.get_all_active_models() if model is not None
        ]
        configuration_client = ConfigurationClient(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        configuration = configuration_client.get_full_configuration()
        if len(active_models) != len(project.get_trainable_tasks()):
            raise ValueError(
                f"Project `{project.name}` does not have a trained model for each "
                f"task in the project. Unable to create deployment, please ensure all "
                f"tasks are trained first."
            )
        deployed_models: List[DeployedModel] = []
        for model_index, model in enumerate(active_models):
            model_config = configuration.task_chain[model_index]
            optimized_models = model.optimized_models
            optimization_types = [
                op_model.optimization_type for op_model in optimized_models
            ]
            preferred_model = optimized_models[0]
            for optimization_type in OptimizationType:
                if optimization_type in optimization_types:
                    preferred_model = optimized_models[
                        optimization_types.index(optimization_type)
                    ]
                    break
            deployed_model = DeployedModel.from_model_and_hypers(
                model=preferred_model, hyper_parameters=model_config
            )
            print(
                f"Retrieving {preferred_model.optimization_type} model data for "
                f"{project.get_trainable_tasks()[model_index].title}..."
            )
            deployed_model.get_data(source=self.session)
            deployed_models.append(deployed_model)
        deployment = Deployment(project=project, models=deployed_models)
        if output_folder is not None:
            deployment.save(output_folder)
        return deployment
