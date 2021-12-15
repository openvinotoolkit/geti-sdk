import os
import warnings
from typing import Optional

from .annotation_readers import SCAnnotationReader
from .rest_managers import (
    ProjectManager,
    MediaManager,
    AnnotationManager,
    ConfigurationManager
)
from .data_models import Project
from .http_session import SCSession, ClusterConfig
from .utils import get_default_workspace_id


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
            self, project_name: str, target_folder: Optional[str] = None
    ) -> Project:
        """
        Download a project with name `project_name` to the local disk. All images and
        image annotations in the project are downloaded.

        This method will download data to the path `target_folder`, the contents will
        be:
            'images'       -- Directory holding all images in the project
            'annotations'  -- Directory holding all annotations in the project, in .json
                              format
            'project.json' -- File containing the project parameters, that can be used
                              to re-create the project.

        :param project_name: Name of the project to download
        :param target_folder: Path to the local folder in which the project data
            should be saved. If not specified, a new directory named `project_name`
            will be created inside the current working directory.
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
        media_manager = MediaManager(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        media_manager.download_all_images(path_to_folder=target_folder)

        # Download annotations
        image_id_mapping = media_manager.get_all_images()
        with warnings.catch_warnings():
            # The AnnotationManager will give a warning that it can only be used to
            # download data since no annotation reader is passed, but this is exactly
            # what we plan to do with it so we suppress the warning
            warnings.simplefilter("ignore")
            annotation_manager = AnnotationManager(
                workspace_id=self.workspace_id,
                session=self.session,
                project=project,
                image_to_id_mapping=image_id_mapping
            )
        annotation_manager.download_all_annotations(path_to_folder=target_folder)
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
        media_manager = MediaManager(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        image_id_mapping = media_manager.upload_folder(
            path_to_folder=os.path.join(target_folder, "images")
        )

        # Disable auto-train to prevent the project from training right away
        configuration_manager = ConfigurationManager(
            workspace_id=self.workspace_id, session=self.session, project=project
        )
        configuration_manager.set_project_auto_train(auto_train=False)

        # Upload annotations for all tasks
        for task_index, task in enumerate(project.get_trainable_tasks()):
            annotation_reader = SCAnnotationReader(
                base_data_folder=os.path.join(target_folder, "annotations"),
                task_type=task.type,
                task_type_to_label_names_mapping={task.type: task.label_names}
            )
            annotation_manager = AnnotationManager[SCAnnotationReader](
                workspace_id=self.workspace_id,
                session=self.session,
                project=project,
                image_to_id_mapping=image_id_mapping,
                annotation_reader=annotation_reader
            )
            print(
                f"Uploading annotations for task number {task_index + 1} of type "
                f"'{task.type}'"
            )
            append_annotations = True
            if task_index == 0:
                append_annotations = False
            annotation_manager.upload_annotations_for_images(
                image_id_list=list(image_id_mapping.values()),
                append_annotations=append_annotations
            )
        configuration_manager.set_project_auto_train(
            auto_train=enable_auto_train
        )
        print(f"Project '{project.name}' was uploaded successfully.")
        return project
