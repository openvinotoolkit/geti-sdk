import copy
import os

from sc_api_tools.rest_managers import (
    AnnotationManager,
    ConfigurationManager,
    MediaManager,
    ProjectManager
)
from sc_api_tools.annotation_readers import DatumAnnotationReader
from sc_api_tools.http_session import SCSession, ClusterConfig
from sc_api_tools import utils


if __name__ == "__main__":

    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Server configuration
    CLUSTER_HOST = 'https://0.0.0.0'
    CLUSTER_USERNAME = "dummy_user@dummy_email.com"
    CLUSTER_PASSWORD = "dummy_password"

    # Dataset configuration
    # Path to the base folder containing the 'images' and 'annotations' folders
    PATH_TO_COCO_DATASET = os.path.join("", "dummy_dataset")
    PATH_TO_IMAGES_IN_COCO_DATASET = os.path.join(  # Path to the actual images
        PATH_TO_COCO_DATASET, "subset", "images"
    )
    NUMBER_OF_IMAGES_TO_UPLOAD = 50

    # Label to filter the dataset on. Can contain only one label for a pipeline project
    LABELS_OF_INTEREST = ["dog"]

    # Project configuration

    # Current options for project_type are 'detection', 'segmentation' or
    # 'detection_to_segmentation'
    PROJECT_TYPE = "detection_to_segmentation"
    PROJECT_NAME = "COCO dog detection and segmentation"

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
    label_names_segmentation = utils.generate_segmentation_labels(label_names)

    # Create or get project
    workspace_id = utils.get_default_workspace_id(session)
    project_manager = ProjectManager(session=session, workspace_id=workspace_id)
    project = project_manager.get_or_create_project(
        project_name=PROJECT_NAME,
        project_type=PROJECT_TYPE,
        labels=[label_names, label_names_segmentation]
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

    # Annotations for detection task
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
    # Upload detection annotations
    annotation_manager.upload_annotations_for_images(list(image_id_mapping.values()))

    # Prepare annotations for segmentation task
    segmentation_label_map = copy.deepcopy(annotation_reader.datum_label_map)
    detection_index = segmentation_label_map.pop(label_names[0])
    segmentation_label_map.update({label_names_segmentation[0]: detection_index})
    annotation_reader.override_label_map(segmentation_label_map)
    annotation_reader.prepare_and_set_dataset(
        task_type=project.get_trainable_tasks()[1].type
    )
    # Upload segmentation annotations
    annotation_manager.upload_annotations_for_images(
        list(image_id_mapping.values()), append_annotations=True
    )
    configuration_manager.set_project_auto_train(auto_train=AUTO_TRAIN_AFTER_UPLOAD)
