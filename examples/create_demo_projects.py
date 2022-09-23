from dotenv import dotenv_values

from geti_sdk import Geti
from geti_sdk.demos import (
    create_anomaly_classification_demo_project,
    create_classification_demo_project,
    create_detection_demo_project,
    create_detection_to_classification_demo_project,
    create_detection_to_segmentation_demo_project,
    create_segmentation_demo_project,
    get_coco_dataset,
)

if __name__ == "__main__":
    # Get credentials from .env file
    env_variables = dotenv_values(dotenv_path=".env")

    if not env_variables:
        raise ValueError(
            "Unable to load login details from .env file, please make sure the file "
            "exists at the root of the `examples` directory."
        )

    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------
    # Set up the Geti instance with server address and login details
    geti = Geti(
        host=env_variables.get("HOST"),
        username=env_variables.get("USERNAME"),
        password=env_variables.get("PASSWORD"),
    )

    # Dataset configuration
    NUMBER_OF_IMAGES_TO_UPLOAD = 100
    NUMBER_OF_IMAGES_TO_ANNOTATE = 75

    # Set this to True if you want the projects to start auto-training after the
    # annotations have been uploaded. Set to False to disable auto-training for the
    # moment
    AUTO_TRAIN_AFTER_UPLOAD = False

    # If you already have the COCO data downloaded on your system, you can point the
    # `COCO_PATH` to the folder containing it. If you leave the COCO_PATH as None,
    # the script will attempt to download the data, or use the dataset from the
    # default path if it has been downloaded before.
    COCO_PATH = None

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------

    # Check that the MS COCO dataset is found at the path, download otherwise
    get_coco_dataset(COCO_PATH, verbose=True)

    # Create the demo projects
    create_segmentation_demo_project(
        geti=geti,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=NUMBER_OF_IMAGES_TO_ANNOTATE,
        auto_train=AUTO_TRAIN_AFTER_UPLOAD,
        dataset_path=COCO_PATH,
    )
    create_detection_demo_project(
        geti=geti,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=NUMBER_OF_IMAGES_TO_ANNOTATE,
        auto_train=AUTO_TRAIN_AFTER_UPLOAD,
        dataset_path=COCO_PATH,
    )
    create_classification_demo_project(
        geti=geti,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=NUMBER_OF_IMAGES_TO_ANNOTATE,
        auto_train=AUTO_TRAIN_AFTER_UPLOAD,
        dataset_path=COCO_PATH,
    )
    create_anomaly_classification_demo_project(
        geti=geti,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=-1,
    )
    create_detection_to_segmentation_demo_project(
        geti=geti,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=NUMBER_OF_IMAGES_TO_ANNOTATE,
        auto_train=AUTO_TRAIN_AFTER_UPLOAD,
        dataset_path=COCO_PATH,
    )
    create_detection_to_classification_demo_project(
        geti=geti,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=NUMBER_OF_IMAGES_TO_ANNOTATE,
        auto_train=AUTO_TRAIN_AFTER_UPLOAD,
        dataset_path=COCO_PATH,
    )
