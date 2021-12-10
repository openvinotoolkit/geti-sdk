import os

from sc_api_tools import (
    AnnotationManager,
    MediaManager,
    ProjectManager,
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
    SERVER_USERNAME = "dummy_username"
    SERVER_PASSWORD = "dummy_password"

    # Path to target folder for download. The download script will create this folder
    # if it does not exist, and will create the directories "images" and
    # "annotations" inside this folder
    TARGET_FOLDER = os.path.join(".", "dummy_project")

    # Project configuration
    PROJECT_NAME = "dummy project"

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

    # Get project and workspace
    workspace_id = get_default_workspace_id(session)
    project_manager = ProjectManager(session=session, workspace_id=workspace_id)
    project = project_manager.get_project_by_name(PROJECT_NAME)

    # Download project creation parameters:
    project_manager.download_project_parameters(
        project_name=PROJECT_NAME, path_to_folder=TARGET_FOLDER
    )

    # Download images
    media_manager = MediaManager(
        workspace_id=workspace_id, session=session, project=project
    )
    media_manager.download_all_images(path_to_folder=TARGET_FOLDER)

    # Download annotations
    image_id_mapping = media_manager.get_all_images()
    annotation_manager = AnnotationManager(
        workspace_id=workspace_id,
        session=session,
        project=project,
        image_to_id_mapping=image_id_mapping
    )
    annotation_manager.download_all_annotations(path_to_folder=TARGET_FOLDER)
