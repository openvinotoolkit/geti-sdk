import os

from sc_api_tools import SCRESTClient
from sc_api_tools.annotation_readers import DatumAnnotationReader


if __name__ == "__main__":
    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Set up REST client with server address and login details
    client = SCRESTClient(
        host="https://0.0.0.0", username="dummy_user", password="dummy_password"
    )

    # Dataset configuration
    # Path to the base folder containing the 'images' and 'annotations' folders
    PATH_TO_COCO_DATASET = os.path.join("", "dummy_dataset")
    PATH_TO_IMAGES_IN_COCO_DATASET = os.path.join(  # Path to the actual images
        PATH_TO_COCO_DATASET, "subset", "images"
    )
    NUMBER_OF_IMAGES_TO_UPLOAD = 50
    
    # Labels to filter the dataset on. Only images containing this object will be
    # uploaded and annotated. Multiple labels can be passed, and the way the dataset
    # is filtered can be configured. See the 'Create annotation reader' section below
    LABELS_OF_INTEREST = ["dog"]

    # Project configuration

    # Current options for project_type are 'detection', 'segmentation' or
    # 'classification'
    PROJECT_TYPE = "segmentation"
    PROJECT_NAME = "COCO dog segmentation"

    # Change this to True if you want the project to start auto-training after the
    # annotations have been uploaded
    AUTO_TRAIN_AFTER_UPLOAD = False

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------
    # Create annotation reader
    annotation_reader = DatumAnnotationReader(
        base_data_folder=PATH_TO_COCO_DATASET,
        annotation_format='coco'
    )
    annotation_reader.filter_dataset(
        labels=LABELS_OF_INTEREST, criterion='OR'
    )
    # Create project and upload data
    client.create_single_task_project_from_dataset(
        project_name=PROJECT_NAME,
        project_type=PROJECT_TYPE,
        path_to_images=PATH_TO_IMAGES_IN_COCO_DATASET,
        annotation_reader=annotation_reader,
        labels=LABELS_OF_INTEREST,
        number_of_images_to_upload=NUMBER_OF_IMAGES_TO_UPLOAD,
        enable_auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )
