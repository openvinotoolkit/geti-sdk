import os

from sc_api_tools import SCRESTClient
from sc_api_tools.annotation_readers import VitensAnnotationReader

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
    PATH_TO_DATASET = os.path.join("", "dummy_dataset")
    NUMBER_OF_IMAGES_TO_UPLOAD = 12

    # Project configuration
    # Current options for project_type are 'detection' and 'segmentation' for Vitens
    # dataset
    PROJECT_TYPE = "detection"
    PROJECT_NAME = "Vitens Aeromonas detection"

    # Change this to True if you want the project to start auto-training after the
    # annotations have been uploaded
    AUTO_TRAIN_AFTER_UPLOAD = False

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------
    # Create annotation reader
    annotation_reader = VitensAnnotationReader(
        base_data_folder=os.path.join(PATH_TO_DATASET, "annotation")
    )
    # Create project and upload data
    client.create_single_task_project_from_dataset(
        project_name=PROJECT_NAME,
        project_type=PROJECT_TYPE,
        path_to_images=os.path.join(PATH_TO_DATASET, "images"),
        annotation_reader=annotation_reader,
        number_of_images_to_upload=NUMBER_OF_IMAGES_TO_UPLOAD,
        enable_auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )
