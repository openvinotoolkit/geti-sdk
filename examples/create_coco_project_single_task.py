from geti_sdk import Geti
from geti_sdk.annotation_readers import DatumAnnotationReader
from geti_sdk.demos import get_coco_dataset
from geti_sdk.utils import get_server_details_from_env

if __name__ == "__main__":
    # Get the server configuration from .env file
    server_config = get_server_details_from_env()

    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Set up the Geti instance with the server configuration details
    geti = Geti(server_config=server_config)

    # Dataset configuration
    NUMBER_OF_IMAGES_TO_UPLOAD = 75
    NUMBER_OF_IMAGES_TO_ANNOTATE = 50

    # Labels to filter the dataset on. Only images containing this object will be
    # uploaded and annotated. Multiple labels can be passed, and the way the dataset
    # is filtered can be configured. See the 'Create annotation reader' section below
    LABELS_OF_INTEREST = ["dog"]

    # Project configuration

    # Current options for project_type are 'detection', 'segmentation' or
    # 'classification'
    PROJECT_TYPE = "detection"
    PROJECT_NAME = "COCO dog detection"

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

    # Create annotation reader
    annotation_reader = DatumAnnotationReader(
        base_data_folder=coco_path, annotation_format="coco"
    )
    annotation_reader.filter_dataset(labels=LABELS_OF_INTEREST, criterion="OR")
    # Create project and upload data
    geti.create_single_task_project_from_dataset(
        project_name=PROJECT_NAME,
        project_type=PROJECT_TYPE,
        path_to_images=coco_path,
        annotation_reader=annotation_reader,
        labels=LABELS_OF_INTEREST,
        number_of_images_to_upload=NUMBER_OF_IMAGES_TO_UPLOAD,
        number_of_images_to_annotate=NUMBER_OF_IMAGES_TO_ANNOTATE,
        enable_auto_train=AUTO_TRAIN_AFTER_UPLOAD,
    )
