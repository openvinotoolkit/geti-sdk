from sc_api_tools import (
    AnnotationManager,
    ConfigurationManager,
    MediaManager,
    ProjectManager,
    DatumAnnotationReader,
    SCSession,
    ServerConfig,
    get_default_workspace_id
)

if __name__ == "__main__":
    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Server configuration
    SERVER_IP = 'https://0.0.0.0'
    SERVER_USERNAME = "dummy_user@dummy_email.com"
    SERVER_PASSWORD = "dummy_password"

    # Dataset configuration
    # Path to the base folder containing the 'images' and 'annotations' folders
    PATH_TO_COCO_DATASET = ""
    PATH_TO_IMAGES_IN_COCO_DATASET = ""  # Path to the actual images
    NUMBER_OF_IMAGES_TO_UPLOAD = 50

    # Labels to filter the dataset on. Only images containing this object will be
    # uploaded and annotated. Multiple labels can be passed, see the 'Prepare dataset'
    # section below for details on the filtering
    LABELS_OF_INTEREST = ["dog"]

    # Project configuration

    # Current options for project_type are 'detection', 'segmentation' or
    # 'detection_to_segmentation'
    PROJECT_TYPE = "detection"
    PROJECT_NAME = "COCO dog detection"

    # Change this to True if you want the project to start auto-training after the
    # annotations have been uploaded
    AUTO_TRAIN_AFTER_UPLOAD = False

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

    # Prepare dataset
    annotation_reader = DatumAnnotationReader(
        base_data_folder=PATH_TO_COCO_DATASET,
        annotation_format="coco",
        task_type=ProjectManager.get_task_types_by_project_type(PROJECT_TYPE)[0]
    )
    annotation_reader.filter_dataset(
        labels=LABELS_OF_INTEREST, criterion="OR"
    )
    image_list = annotation_reader.get_all_image_names()
    label_names = annotation_reader.get_all_label_names()

    # Create or get project
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
    configuration_manager.set_project_num_iterations(value=100)

    # Upload images
    media_manager = MediaManager(
        session=session, workspace_id=workspace_id, project=project
    )
    image_id_mapping = media_manager.upload_from_list(
        path_to_folder=PATH_TO_IMAGES_IN_COCO_DATASET,
        image_names=image_list,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD
    )

    # Upload annotations
    annotation_reader.prepare_and_set_dataset(
        task_type=project.get_trainable_tasks()[0].type
    )
    annotation_manager = AnnotationManager[DatumAnnotationReader](
        session=session,
        workspace_id=workspace_id,
        project=project,
        annotation_reader=annotation_reader,
        image_to_id_mapping=image_id_mapping
    )
    # Annotations for detection task
    annotation_manager.upload_annotations_for_images(list(image_id_mapping.values()))
    configuration_manager.set_project_auto_train(auto_train=AUTO_TRAIN_AFTER_UPLOAD)
