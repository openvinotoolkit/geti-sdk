import os

from rest_managers import (
    AnnotationManager,
    ConfigurationManager,
    MediaManager,
    ProjectManager
)
from annotation_readers import DatumAnnotationReader
from http_session import SCSession, ServerConfig
from utils import get_default_workspace_id


if __name__ == "__main__":

    bare_metal_ip = "https://10.91.120.238/"
    vm_ip = "https://10.55.252.51"

    config = ServerConfig(
        host=bare_metal_ip,
        username="admin@sc-project.intel.com",
        password="admin"
    )

    session = SCSession(serverconfig=config)

    base_data_path = os.path.join(
        "C:\\", "Users", "ljcornel", "OneDrive - Intel Corporation", "Documents",
        "Datasets"
    )
    data_path = os.path.join(base_data_path, "COCO")
    image_data_path = os.path.join(data_path, "images", "val2017")

    annotation_reader = DatumAnnotationReader(
        base_data_folder=data_path, annotation_format="coco", task_type="segmentation"
    )
    annotation_reader.filter_dataset(
        labels=["dog"], criterion="AND"
    )
    annotation_reader.prepare_and_set_dataset(task_type="segmentation")

    image_list = annotation_reader.get_all_image_names()
    label_names = annotation_reader.get_all_label_names()

    # Create or get project
    workspace_id = get_default_workspace_id(session)
    project_manager = ProjectManager(session=session, workspace_id=workspace_id)
    project = project_manager.get_or_create_project(
        project_name="Test coco dog segmentation",
        project_type="segmentation",
        label_names_task_one=label_names
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
    image_id_mapping = media_manager.upload_from_list(
        path_to_folder=image_data_path, image_names=image_list
    )

    # Upload annotations
    annotation_manager = AnnotationManager[DatumAnnotationReader](
        session=session,
        workspace_id=workspace_id,
        project=project,
        annotation_reader=annotation_reader,
        image_to_id_mapping=image_id_mapping
    )
    # Annotations for segmentation task
    annotation_manager.upload_annotations_for_images(list(image_id_mapping.values()))
