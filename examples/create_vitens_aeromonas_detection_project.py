import os

from sc_api_tools.rest_managers import (
    AnnotationManager,
    ConfigurationManager,
    MediaManager,
    ProjectManager
)
from sc_api_tools.annotation_readers import VitensAnnotationReader
from sc_api_tools.http_session import SCSession, ClusterConfig
from sc_api_tools.utils import get_default_workspace_id


if __name__ == "__main__":

    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Server configuration
    CLUSTER_HOST = "https://0.0.0.0"
    CLUSTER_USERNAME = "dummy_user"
    CLUSTER_PASSWORD = "dummy_password"

    # Dataset configuration
    # Path to the base folder containing the 'images' and 'annotations' folders
    PATH_TO_DATASET = os.path.join("", "dummy_dataset")
    NUMBER_OF_IMAGES_TO_UPLOAD = 12

    # Project configuration

    # Current options for project_type are 'detection', 'segmentation' or
    # 'detection_to_segmentation'
    PROJECT_TYPE = "detection"
    PROJECT_NAME = "Vitens Aeromonas detection"

    # Change this to True if you want the project to start auto-training after the
    # annotations have been uploaded
    AUTO_TRAIN_AFTER_UPLOAD = False

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------

    # Initialize http session
    config = ClusterConfig(
        host=CLUSTER_HOST,
        username=CLUSTER_USERNAME,
        password=CLUSTER_PASSWORD
    )
    session = SCSession(config)

    # Dataset information
    annotations_data_path = os.path.join(PATH_TO_DATASET, "annotation")
    image_data_path = os.path.join(PATH_TO_DATASET, "images")

    # Extract label names
    annotation_reader = VitensAnnotationReader(
        base_data_folder=annotations_data_path,
        task_type=ProjectManager.get_task_types_by_project_type(PROJECT_TYPE)[0]
    )
    label_names = annotation_reader.get_all_label_names()

    # Create project
    workspace_id = get_default_workspace_id(session)
    project_manager = ProjectManager(session=session, workspace_id=workspace_id)
    project = project_manager.get_or_create_project(
        project_name=PROJECT_NAME,
        project_type=PROJECT_TYPE,
        labels=[label_names]
    )

    # Disable auto training
    configuration_manager = ConfigurationManager(
        session=session, workspace_id=workspace_id, project=project
    )
    configuration_manager.set_project_auto_train(auto_train=False)

    # Upload images
    media_manager = MediaManager(
        session=session, workspace_id=workspace_id, project=project
    )
    image_id_mapping = media_manager.upload_folder(
        image_data_path, n_images=NUMBER_OF_IMAGES_TO_UPLOAD
    )

    # Upload annotations
    annotation_manager = AnnotationManager[VitensAnnotationReader](
        session=session,
        workspace_id=workspace_id,
        project=project,
        annotation_reader=annotation_reader,
        image_to_id_mapping=image_id_mapping
    )
    annotation_manager.upload_annotations_for_images(list(image_id_mapping.values()))
    configuration_manager.set_project_auto_train(auto_train=AUTO_TRAIN_AFTER_UPLOAD)
