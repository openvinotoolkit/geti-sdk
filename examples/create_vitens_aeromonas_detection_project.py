import os

from sc_api_tools import (
    AnnotationManager,
    ConfigurationManager,
    MediaManager,
    ProjectManager,
    VitensAnnotationReader,
    SCSession,
    ServerConfig,
    get_default_workspace_id
)


if __name__ == "__main__":

    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Server configuration
    SERVER_IP = "https://0.0.0.0"
    SERVER_USERNAME = "dummy_user"
    SERVER_PASSWORD = "dummy_password"

    # Dataset configuration
    # Path to the base folder containing the 'images' and 'annotations' folders
    PATH_TO_DATASET = ""
    NUMBER_OF_IMAGES_TO_UPLOAD = 12

    # Project configuration

    # Current options for project_type are 'detection', 'segmentation' or
    # 'detection_to_segmentation'
    PROJECT_TYPE = "detection"
    PROJECT_NAME = "Vitens Aeromonas detection"

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------

    # Initialize http session
    config = ServerConfig(
        host=SERVER_IP,
        username=SERVER_USERNAME,
        password=SERVER_PASSWORD
    )
    session = SCSession(serverconfig=config)

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
