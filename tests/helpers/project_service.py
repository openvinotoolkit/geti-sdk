from typing import Optional, List, Union, Dict, Any

from sc_api_tools.annotation_readers import AnnotationReader, DatumAnnotationReader
from vcr import VCR

from sc_api_tools import SCRESTClient
from sc_api_tools.data_models import Project
from sc_api_tools.rest_managers import (
    ProjectManager,
    ConfigurationManager,
    ImageManager,
    AnnotationManager,
    TrainingManager,
    VideoManager,
    ModelManager,
    PredictionManager,
)
from .constants import CASSETTE_EXTENSION, PROJECT_PREFIX
from .finalizers import force_delete_project


class ProjectService:
    """
    This class contains functionality to quickly create projects and interact with
    them through the respective manager clients

    :param client: SCRESTClient instance representing the SC server and workspace in
        which to create the project
    :param vcr: VCR instance used for recording HTTP requests made during the project
        lifespan
    """
    def __init__(self, client: SCRESTClient, vcr: VCR):
        self.vcr = vcr
        self.session = client.session
        self.workspace_id = client.workspace_id
        self.project_manager = ProjectManager(
            session=client.session, workspace_id=client.workspace_id
        )

        self._project: Optional[Project] = None
        self._configuration_manager: Optional[ConfigurationManager] = None
        self._image_manager: Optional[ImageManager] = None
        self._annotation_manager: Optional[AnnotationManager] = None
        self._training_manager: Optional[TrainingManager] = None
        self._video_manager: Optional[VideoManager] = None
        self._model_manager: Optional[ModelManager] = None
        self._prediction_manager: Optional[PredictionManager] = None
        self._manager_names = [
            "_configuration_manager",
            "_image_manager",
            "_annotation_manager",
            "_training_manager",
            "_video_manager",
            "_model_manager",
            "_prediction_manager"
        ]

    def create_project(
            self,
            project_name: Optional[str] = None,
            project_type: str = "classification",
            labels: Optional[List[Union[List[str], List[Dict[str, Any]]]]] = None
    ) -> Project:
        """
        Create a project according to the `name`, `project_type` and `labels` specified.

        :param project_name: Name of the project to create
        :param project_type: Type of the project to create
        :param labels: List of labels for each task
        :return: the created project
        """
        if project_name is None:
            project_name = f"{PROJECT_PREFIX}_project_simple"
        if self._project is None:
            if labels is None:
                labels = [["cube", "cylinder"]]
            with self.vcr.use_cassette(f"{project_name}.{CASSETTE_EXTENSION}"):
                project = self.project_manager.create_project(
                    project_name=project_name,
                    project_type=project_type,
                    labels=labels
                )
                self._project = project
                return project
        else:
            raise ValueError(
                "This ProjectService instance already contains an existing project. "
                "Please either delete the existing project first or use a new "
                "instance to create another project"
            )

    def get_or_create_project(
            self,
            project_name: str = "sdk_test_project_simple",
            project_type: str = "classification",
            labels: Optional[List[Union[List[str], List[Dict[str, Any]]]]] = None
    ) -> Project:
        """
        This method will always return a project. It will either create a new one, or
        return the existing project if it has already been created.

        :param project_name: Name of the project to create
        :param project_type: Type of the project to create
        :param labels: List of labels for each task
        :return: the existing or newly created project
        """
        if not self.has_project:
            self.create_project(
                project_name=project_name, project_type=project_type, labels=labels
            )
        return self.project

    @property
    def project(self) -> Project:
        """
        Returns the project managed by the ProjectService.

        :return:
        """
        if self._project is None:
            raise ValueError(
                "This ProjectService instance does not contain a project yet. Please "
                "call `ProjectService.create_project` to create a new project first."
            )
        return self._project

    @property
    def has_project(self) -> bool:
        """
        Returns True if the ProjectService contains an existing project, False otherwise
        """
        return self._project is not None

    @property
    def image_manager(self) -> ImageManager:
        """ Returns the ImageManager instance for the project """
        if self._image_manager is None:
            with self.vcr.use_cassette(
                f"{self.project.name}_image_manager.{CASSETTE_EXTENSION}"
            ):
                self._image_manager = ImageManager(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project
                )
        return self._image_manager

    @property
    def video_manager(self) -> VideoManager:
        """ Returns the VideoManager instance for the project """
        if self._video_manager is None:
            with self.vcr.use_cassette(
                f"{self.project.name}_video_manager.{CASSETTE_EXTENSION}"
            ):
                self._video_manager = VideoManager(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project
                )
        return self._video_manager

    @property
    def annotation_manager(self) -> AnnotationManager:
        """ Returns the AnnotationManager instance for the project """
        if self._annotation_manager is None:
            with self.vcr.use_cassette(
                f"{self.project.name}_annotation_manager.{CASSETTE_EXTENSION}"
            ):
                self._annotation_manager = AnnotationManager(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project
                )
        return self._annotation_manager

    @property
    def configuration_manager(self) -> ConfigurationManager:
        """ Returns the ConfigurationManager instance for the project """
        if self._configuration_manager is None:
            with self.vcr.use_cassette(
                f"{self.project.name}_configuration_manager.{CASSETTE_EXTENSION}"
            ):
                self._configuration_manager = ConfigurationManager(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project
                )
        return self._configuration_manager

    @property
    def training_manager(self) -> TrainingManager:
        """ Returns the TrainingManager instance for the project """
        if self._training_manager is None:
            with self.vcr.use_cassette(
                f"{self.project.name}_training_manager.{CASSETTE_EXTENSION}"
            ):
                self._training_manager = TrainingManager(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project
                )
        return self._training_manager

    @property
    def prediction_manager(self) -> PredictionManager:
        """ Returns the PredictionManager instance for the project """
        if self._prediction_manager is None:
            with self.vcr.use_cassette(
                    f"{self.project.name}_prediction_manager.{CASSETTE_EXTENSION}"
            ):
                self._prediction_manager = PredictionManager(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project
                )
        return self._prediction_manager

    @property
    def model_manager(self) -> ModelManager:
        """ Returns the ModelManager instance for the project """
        if self._model_manager is None:
            with self.vcr.use_cassette(
                    f"{self.project.name}_model_manager.{CASSETTE_EXTENSION}"
            ):
                self._model_manager = ModelManager(
                    session=self.session,
                    workspace_id=self.workspace_id,
                    project=self.project
                )
        return self._model_manager

    def delete_project(self):
        """ Deletes the project from the server """
        if self._project is not None:
            with self.vcr.use_cassette(
                f"{self.project.name}_deletion.{CASSETTE_EXTENSION}"
            ):
                force_delete_project(self.project.name, self.project_manager)
                self.reset_state()

    def reset_state(self) -> None:
        """
        Resets the state of the ProjectService instance. This method should be called
        once the project belonging to the project service is deleted from the server
        """
        self._project = None
        for manager_name in self._manager_names:
            setattr(self, manager_name, None)

    def add_annotated_media(
            self, annotation_reader: AnnotationReader, n_images: int = 12
    ) -> None:
        """
        Adds annotated media to the project

        :param annotation_reader: AnnotationReader to use to get annotations for the
            media
        :param n_images: Number of annotated images to upload. If set to -1, uploads
            all images for which the annotation reader contains annotations
        """
        data_path = annotation_reader.base_folder
        with self.vcr.use_cassette(
            f"{self.project.name}_add_annotated_media.{CASSETTE_EXTENSION}"
        ):
            if isinstance(annotation_reader, DatumAnnotationReader):
                images = self.image_manager.upload_from_list(
                    path_to_folder=data_path,
                    image_names=annotation_reader.get_all_image_names(),
                    n_images=n_images
                )
            else:
                images = self.image_manager.upload_folder(
                    data_path, n_images=n_images
                )

            if n_images < len(images) and n_images != -1:
                images = images[:n_images]
            # Annotation preparation and upload
            for task_index, task in enumerate(self.project.get_trainable_tasks()):
                # Set annotation reader task type
                annotation_reader.task_type = task.type
                annotation_reader.prepare_and_set_dataset(
                    task_type=task.type
                )
                self.annotation_manager.annotation_reader = annotation_reader
                # Upload annotations
                self.annotation_manager.upload_annotations_for_images(
                    images=images,
                    append_annotations=task_index > 0
                )

    def set_auto_train(self, auto_train: bool = True) -> None:
        """
        Sets the 'auto_training' parameter for all tasks in the project to `auto_train`

        :param auto_train: True to turn auto_training on, False to turn it off
        """
        with self.vcr.use_cassette(
            f"{self.project.name}_set_auto_train.{CASSETTE_EXTENSION}"
        ):
            self.configuration_manager.set_project_auto_train(auto_train=auto_train)

    def set_minimal_training_hypers(self) -> None:
        """
        Configures the project to use the lowest possible batch size and number of
        epochs to perform a minimal training round

        """
        with self.vcr.use_cassette(
            f"{self.project.name}_set_minimal_hypers.{CASSETTE_EXTENSION}"
        ):
            self.configuration_manager.set_project_num_iterations(1)
            for task in self.project.get_trainable_tasks():
                task_config = self.configuration_manager.get_task_configuration(task.id)
                try:
                    task_config.set_parameter_value('batch_size', 1)
                    self.configuration_manager.set_configuration(task_config)
                except ValueError:
                    print(
                        f"Parameter batch_size was not found in the configuration for "
                        f"task {task.summary}. Unable to configure batch size"
                    )
