from sc_api_tools import SCRESTClient

from sc_api_tools.demos import (
    create_segmentation_demo_project,
    create_detection_demo_project,
    create_classification_demo_project,
    create_anomaly_classification_demo_project,
    create_detection_to_segmentation_demo_project,
    create_detection_to_classification_demo_project
)


if __name__ == "__main__":
    # --------------------------------------------------
    # Configuration section
    # --------------------------------------------------

    # Set up REST client with server address and login details
    client = SCRESTClient(
        host="https://0.0.0.0", username="dummy_user", password="dummy_password"
    )

    # Dataset configuration
    NUMBER_OF_IMAGES_TO_UPLOAD = 100
    NUMBER_OF_IMAGES_TO_ANNOTATE = 75

    # Change this to True if you want the projects to start auto-training after the
    # annotations have been uploaded
    AUTO_TRAIN_AFTER_UPLOAD = False

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------

    create_segmentation_demo_project(
        client=client,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=NUMBER_OF_IMAGES_TO_ANNOTATE,
        auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )

    create_detection_demo_project(
        client=client,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=NUMBER_OF_IMAGES_TO_ANNOTATE,
        auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )

    create_classification_demo_project(
        client=client,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=NUMBER_OF_IMAGES_TO_ANNOTATE,
        auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )

    create_anomaly_classification_demo_project(
        client=client,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=NUMBER_OF_IMAGES_TO_ANNOTATE,
        auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )

    create_detection_to_segmentation_demo_project(
        client=client,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=NUMBER_OF_IMAGES_TO_ANNOTATE,
        auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )

    create_detection_to_classification_demo_project(
        client=client,
        n_images=NUMBER_OF_IMAGES_TO_UPLOAD,
        n_annotations=NUMBER_OF_IMAGES_TO_ANNOTATE,
        auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )
