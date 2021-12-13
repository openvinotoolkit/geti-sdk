import os

from sc_api_tools import (
    AnnotationManager,
    ConfigurationManager,
    MediaManager,
    ProjectManager,
    SCSession,
    ClusterConfig,
    SCAnnotationReader,
    get_default_workspace_id
)

if __name__ == "__main__":

    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Server configuration
    CLUSTER_HOST = "https://0.0.0.0"
    CLUSTER_USERNAME = "dummy_username"
    CLUSTER_PASSWORD = "dummy_password"

    # Path to the folder containing the (previously downloaded) project data. This
    # folder should contain:
    #   - `images` directory containing the project images
    #   - `annotations` directory containing the project annotations
    #   - `project_info.json` file holding the basic project parameters
    TARGET_FOLDER = os.path.join(".", "dummy_project")

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

    # Get project and workspace
    workspace_id = get_default_workspace_id(session)
    project_manager = ProjectManager(session=session, workspace_id=workspace_id)
    project = project_manager.create_project_from_folder(TARGET_FOLDER)

    # Upload images
    media_manager = MediaManager(
        workspace_id=workspace_id, session=session, project=project
    )
    image_id_mapping = media_manager.upload_folder(
        path_to_folder=os.path.join(TARGET_FOLDER, "images")
    )

    # Disable auto-train to prevent the project from training right away
    configuration_manager = ConfigurationManager(
        workspace_id=workspace_id, session=session, project=project
    )
    configuration_manager.set_project_auto_train(auto_train=False)

    # Upload annotations for all tasks
    for task_index, task in enumerate(project.get_trainable_tasks()):
        annotation_reader = SCAnnotationReader(
            base_data_folder=os.path.join(TARGET_FOLDER, "annotations"),
            task_type=task.type,
            task_type_to_label_names_mapping={task.type: task.label_names}
        )
        annotation_manager = AnnotationManager[SCAnnotationReader](
            workspace_id=workspace_id,
            session=session,
            project=project,
            image_to_id_mapping=image_id_mapping,
            annotation_reader=annotation_reader
        )
        print(
            f"Uploading annotations for task number {task_index+1} of type "
            f"'{task.type}'"
        )
        annotation_manager.upload_annotations_for_images(
            image_id_list=list(image_id_mapping.values()), append_annotations=True
        )
    configuration_manager.set_project_auto_train(auto_train=AUTO_TRAIN_AFTER_UPLOAD)
