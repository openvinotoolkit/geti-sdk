import os
import warnings
from typing import Optional, List, Union

from .annotation_readers import (
    SCAnnotationReader,
    AnnotationReader,
    DatumAnnotationReader
)
from .rest_managers import (
    ProjectManager,
    AnnotationManager,
    ConfigurationManager,
    ImageManager,
    VideoManager,
    PredictionManager
)
from .data_models import (
    Project,
    TaskType,
    MediaList,
    Image,
    Video
)
from .http_session import SCSession, ClusterConfig
from .utils import (
    get_default_workspace_id,
    generate_classification_labels,
    get_task_types_by_project_type
)


class SCRESTClient:
    """
    This class is a client to interact with a Sonoma Creek cluster via the REST
    API. It provides methods for project creation, downloading and uploading.

    :param host: IP address or URL at which the cluster can be reached, for example
        'https://0.0.0.0' or 'https://sc_example.intel.com'
    :param username: Username to log in to the cluster
    :param password: Password to log in to the cluster
    :param workspace_id: Optional ID of the workspace that should be addressed by this
        SCRESTClient instance. If not specified, the default workspace is used.
    """
    def __init__(
            self,
            host: str,
            username: str,
            password: str,
            workspace_id: Optional[str] = None
    ):
        self.session = SCSession(
            cluster_config=ClusterConfig(
                host=host, username=username, password=password)
        )
        if workspace_id is None:
            workspace_id = get_default_workspace_id(self.session)
        self.workspace_id = workspace_id

    def download_project(
            self,
            project_name: str,
            target_folder: Optional[str] = None,
            include_predictions: bool = True
    ) -> Project:
        """
        Download a project with name `project_name` to the local disk. All images and
        image annotations in the project are downloaded.

        This method will download data to the path `target_folder`, the contents will
        be:
            'images'       -- Directory holding all images in the project
            'videos'       -- Directory holding all videos in the project
            'annotations'  -- Directory holding all annotations in the project, in .json
                              format
            'predictions'  -- Directory holding all predictions in the project, in
                              .json format. If available, this will include saliency
                              maps in .jpeg format.
            'project.json' -- File containing the project parameters, that can be used
                              to re-create the project.

        :param project_name: Name of the project to download
        :param target_folder: Path to the local folder in which the project data
            should be saved. If not specified, a new directory named `project_name`
            will be created inside the current working directory.
        :param include_predictions: True to also download the predictions for all
            images and videos in the project, False to not download any predictions.
            If this is set to True but the project has no trained models, downloading
            predictions will be skipped.
        :return: Project object, holding information obtained from the cluster
            regarding the downloaded project
        """
        # Obtain project details from cluster
        project_manager = ProjectManager(
            session=self.session, workspace_id=self.workspace_id
        )
        project = project_manager.get_project_by_name(project_name)

        # Validate or create target_folder
        if target_folder is None:
            target_folder = os.path.join('.', project_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Download project creation parameters:
        project_manager.download_project_info(
            project_name=project_name, path_to_folder=target_folder
        )

        # Download images
        image_manager = ImageManager(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        images = image_manager.get_all_images()
        if len(images) > 0:
            image_manager.download_all(path_to_folder=target_folder)

        # Download videos
        video_manager = VideoManager(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        videos = video_manager.get_all_videos()
        if len(videos) > 0:
            video_manager.download_all(path_to_folder=target_folder)

        # Download annotations
        with warnings.catch_warnings():
            # The AnnotationManager will give a warning that it can only be used to
            # download data since no annotation reader is passed, but this is exactly
            # what we plan to do with it so we suppress the warning
            warnings.simplefilter("ignore")
            annotation_manager = AnnotationManager(
                session=self.session,
                project=project,
                media_lists=[images, videos]
            )
        annotation_manager.download_all_annotations(path_to_folder=target_folder)

        # Download predictions
        prediction_manager = PredictionManager(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        if prediction_manager.project_ready and include_predictions:
            if len(images) > 0:
                prediction_manager.download_predictions_for_images(
                    images=images,
                    path_to_folder=target_folder,
                    include_result_media=True
                )
            if len(videos) > 0:
                prediction_manager.download_predictions_for_videos(
                    videos=videos,
                    path_to_folder=target_folder,
                    include_result_media=True,
                    inferred_frames_only=False
                )

        print(f"Project '{project.name}' was downloaded successfully.")
        return project

    def upload_project(
            self,
            target_folder: str,
            project_name: Optional[str] = None,
            enable_auto_train: bool = True
    ) -> Project:
        """
        Upload a previously downloaded SC project to the cluster. This method expects
        the `target_folder` to contain the following:

            'images'       -- Directory holding all images in the project
            'annotations'  -- Directory holding all annotations in the project, in .json
                              format
            'project.json' -- File containing the project parameters, that can be used
                              to re-create the project.

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
        project_manager = ProjectManager(
            session=self.session, workspace_id=self.workspace_id
        )
        project = project_manager.create_project_from_folder(
            path_to_folder=target_folder, project_name=project_name
        )

        # Upload images
        image_manager = ImageManager(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        images = image_manager.upload_folder(
            path_to_folder=os.path.join(target_folder, "images")
        )

        # Upload videos
        video_manager = VideoManager(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        videos = video_manager.upload_folder(
            path_to_folder=os.path.join(target_folder, "videos")
        )

        # Disable auto-train to prevent the project from training right away
        configuration_manager = ConfigurationManager(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        configuration_manager.set_project_auto_train(auto_train=False)

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
        annotation_manager = AnnotationManager[SCAnnotationReader](
            session=self.session,
            project=project,
            media_lists=media_lists,
            annotation_reader=annotation_reader
        )
        if len(images) > 0:
            annotation_manager.upload_annotations_for_images(
                images=images,
            )
        if len(videos) > 0:
            annotation_manager.upload_annotations_for_videos(
                videos=videos,
            )

        configuration_manager.set_project_auto_train(
            auto_train=enable_auto_train
        )
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
            enable_auto_train: bool = True
    ) -> Project:
        """
        This method creates a single task project named `project_name` on the SC
        cluster, and uploads data from a dataset on local disk.

        The type of task that will be in the project can be controlled by setting the
        `project_type`, options are:
            'classification', 'detection', 'segmentation', 'anomaly_classification'

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
            if project_type == 'classification':
                # Handle label generation for classification case
                filter_settings = annotation_reader.applied_filters
                criterion = filter_settings[0]['criterion']
                multilabel = True
                if criterion == 'XOR':
                    multilabel = False
                labels = generate_classification_labels(labels, multilabel=multilabel)
            elif project_type == 'anomaly_classification':
                labels = ["Normal", "Anomalous"]

        # Create project
        project_manager = ProjectManager(
            session=self.session, workspace_id=self.workspace_id
        )
        project = project_manager.get_or_create_project(
            project_name=project_name,
            project_type=project_type,
            labels=[labels]
        )
        # Disable auto training
        configuration_manager = ConfigurationManager(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        configuration_manager.set_project_auto_train(auto_train=False)

        # Upload images
        image_manager = ImageManager(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        if isinstance(annotation_reader, DatumAnnotationReader):
            images = image_manager.upload_from_list(
                path_to_folder=path_to_images,
                image_names=annotation_reader.get_all_image_names(),
                n_images=number_of_images_to_upload
            )
        else:
            images = image_manager.upload_folder(
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
        annotation_manager = AnnotationManager(
            session=self.session,
            project=project,
            annotation_reader=annotation_reader,
            media_lists=[images]
        )
        annotation_manager.upload_annotations_for_images(
            images
        )
        configuration_manager.set_project_auto_train(auto_train=enable_auto_train)
        return project

    def create_task_chain_project_from_dataset(
            self,
            project_name: str,
            project_type: str,
            path_to_images: str,
            label_source_per_task: List[Union[AnnotationReader, List[str]]],
            number_of_images_to_upload: int = -1,
            number_of_images_to_annotate: int = -1,
            enable_auto_train: bool = True
    ) -> Project:
        """
        This method creates a single task project named `project_name` on the SC
        cluster, and uploads data from a dataset on local disk.

        The type of task that will be in the project can be controlled by setting the
        `project_type`, current options are:
            'detection_to_segmentation', 'detection_to_classification'

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
            if isinstance(entry, AnnotationReader) else entry
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
        project_manager = ProjectManager(
            session=self.session, workspace_id=self.workspace_id
        )
        project = project_manager.get_or_create_project(
            project_name=project_name,
            project_type=project_type,
            labels=labels_per_task
        )
        # Disable auto training
        configuration_manager = ConfigurationManager(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        configuration_manager.set_project_auto_train(auto_train=False)

        # Upload images
        image_manager = ImageManager(
            session=self.session, workspace_id=self.workspace_id, project=project
        )
        # Assume that the first task determines the media that will be uploaded
        first_task_reader = annotation_readers_per_task[0]
        if isinstance(first_task_reader, DatumAnnotationReader):
            images = image_manager.upload_from_list(
                path_to_folder=path_to_images,
                image_names=first_task_reader.get_all_image_names(),
                n_images=number_of_images_to_upload
            )
        else:
            images = image_manager.upload_folder(
                path_to_images, n_images=number_of_images_to_upload
            )

        if (
                number_of_images_to_annotate < len(images)
                and number_of_images_to_annotate != -1
        ):
            images = images[:number_of_images_to_annotate]

        append_annotations = False
        for task_type, reader in zip(task_types, annotation_readers_per_task):
            if reader is not None:
                # Set annotation reader task type
                reader.task_type = task_type
                reader.prepare_and_set_dataset(task_type=task_type)
                # Upload annotations
                annotation_manager = AnnotationManager(
                    session=self.session,
                    project=project,
                    annotation_reader=reader,
                    media_lists=[images]
                )
                annotation_manager.upload_annotations_for_images(
                    images=images,
                    append_annotations=append_annotations
                )
                append_annotations = True
        configuration_manager.set_project_auto_train(auto_train=enable_auto_train)
        return project

    @staticmethod
    def _check_unique_label_names(
            labels_per_task: List[List[str]],
            task_types: List[TaskType],
            annotation_readers_per_task: List[AnnotationReader]
    ):
        """
        Checks that the names of all labels passed in `labels_per_task` are unique. If
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
        Downloads all projects in the default workspace from the SC cluster

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
        project_manager = ProjectManager(
            session=self.session, workspace_id=self.workspace_id
        )
        projects = project_manager.get_all_projects()

        # Validate or create target_folder
        if target_folder is None:
            target_folder = os.path.join('.', 'projects')
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
                include_predictions=include_predictions
            )
        return projects

    def upload_all_projects(self, target_folder: str) -> List[Project]:
        """
        Uploads all projects found in the directory `target_folder` on local disk to
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
            folder for folder in candidate_project_folders
            if ProjectManager.is_project_dir(folder)
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
