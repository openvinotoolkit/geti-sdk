import os

from sc_api_tools import SCRESTClient
from sc_api_tools.annotation_readers import DatumAnnotationReader
from sc_api_tools.utils import get_task_types_by_project_type, get_coco_dataset

if __name__ == "__main__":
    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Set up REST client with server address and login details
    client = SCRESTClient(
        host="https://0.0.0.0", username="dummy_user", password="dummy_password"
    )

    # Dataset configuration
    NUMBER_OF_IMAGES_TO_UPLOAD = 75
    NUMBER_OF_IMAGES_TO_ANNOTATE = 50

    # Label to filter the dataset on. Can contain only one label for a pipeline project
    LABELS_OF_INTEREST = ["dog"]

    # Project configuration

    # Currently the only option to create a COCO task chain project is
    # 'detection_to_segmentation'
    PROJECT_TYPE = "detection_to_segmentation"
    PROJECT_NAME = "COCO dog detection and segmentation"

    # Change this to True if you want the project to start auto-training after the
    # annotations have been uploaded
    AUTO_TRAIN_AFTER_UPLOAD = False

    # If you already have the COCO data downloaded on your system, you can point the
    # `COCO_PATH` to the folder containing it. If you leave the COCO_PATH as None,
    # the script will attempt to download the data, or use the dataset from the
    # default path if it has been downloaded before.
    COCO_PATH = None

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------
    coco_path = get_coco_dataset(COCO_PATH)

    # Create annotation readers and apply filters. Use Datumaro annotations for both
    # tasks
    label_source_per_task = []
    for task_type in get_task_types_by_project_type(PROJECT_TYPE):
        annotation_reader = DatumAnnotationReader(
            base_data_folder=coco_path,
            annotation_format='coco',
            task_type=task_type
        )
        annotation_reader.filter_dataset(labels=LABELS_OF_INTEREST, criterion='OR')
        label_source_per_task.append(annotation_reader)

    # Create project and upload data
    client.create_task_chain_project_from_dataset(
        project_name=PROJECT_NAME,
        project_type=PROJECT_TYPE,
        path_to_images=coco_path,
        label_source_per_task=label_source_per_task,
        number_of_images_to_upload=NUMBER_OF_IMAGES_TO_UPLOAD,
        number_of_images_to_annotate=NUMBER_OF_IMAGES_TO_ANNOTATE,
        enable_auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )
