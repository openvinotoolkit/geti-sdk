import os

from sc_api_tools import SCRESTClient
from sc_api_tools.annotation_readers import DatumAnnotationReader
from sc_api_tools.utils import get_task_types_by_project_type

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
    PATH_TO_COCO_DATASET = "data"
    PATH_TO_IMAGES_IN_COCO_DATASET = os.path.join(  # Path to the actual images
        PATH_TO_COCO_DATASET, "images", "val2017"
    )

    NUMBER_OF_IMAGES_TO_UPLOAD = 100
    NUMBER_OF_IMAGES_TO_ANNOTATE = 75

    # Change this to True if you want the project to start auto-training after the
    # annotations have been uploaded
    AUTO_TRAIN_AFTER_UPLOAD = True

    # --------------------------------------------------
    # End of configuration section
    # --------------------------------------------------
    print("\n ------- Creating segmentation project --------------- \n")

    LABELS_OF_INTEREST = ["dog", "person"]
    PROJECT_TYPE = "segmentation"
    PROJECT_NAME = "Segmentation demo"

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
        number_of_images_to_annotate=NUMBER_OF_IMAGES_TO_ANNOTATE,
        enable_auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )

    print("\n ------- Creating detection project --------------- \n")

    LABELS_OF_INTEREST = ["horse", "cat"]
    PROJECT_TYPE = "detection"
    PROJECT_NAME = "Detection demo"

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
        number_of_images_to_annotate=NUMBER_OF_IMAGES_TO_ANNOTATE,
        enable_auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )

    print("\n ------- Creating classification project --------------- \n")

    LABELS_OF_INTEREST = ["horse", "cat", "dog"]
    PROJECT_TYPE = "classification"
    PROJECT_NAME = "Classification demo"

    # Create annotation reader
    annotation_reader = DatumAnnotationReader(
        base_data_folder=PATH_TO_COCO_DATASET,
        annotation_format='coco'
    )
    annotation_reader.filter_dataset(
        labels=LABELS_OF_INTEREST, criterion='XOR'
    )
    # Create project and upload data
    client.create_single_task_project_from_dataset(
        project_name=PROJECT_NAME,
        project_type=PROJECT_TYPE,
        path_to_images=PATH_TO_IMAGES_IN_COCO_DATASET,
        annotation_reader=annotation_reader,
        labels=LABELS_OF_INTEREST,
        number_of_images_to_upload=NUMBER_OF_IMAGES_TO_UPLOAD,
        number_of_images_to_annotate=NUMBER_OF_IMAGES_TO_ANNOTATE,
        enable_auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )

    print("\n ------- Creating anomaly classification project --------------- \n")

    LABELS_OF_INTEREST = ["horse", "dog", "cat", "car", "bicycle"]
    PROJECT_TYPE = "anomaly_classification"
    PROJECT_NAME = "Anomaly classification demo"

    # Create annotation reader
    annotation_reader = DatumAnnotationReader(
        base_data_folder=PATH_TO_COCO_DATASET,
        annotation_format='coco'
    )
    annotation_reader.filter_dataset(
        labels=LABELS_OF_INTEREST, criterion='XOR'
    )

    # Create groups of "Normal" and "Anomalous" images, since those are the only
    # label names allowed for anomaly classification
    # All animals will be considered "Normal"
    annotation_reader.group_labels(
        labels_to_group=["horse", "dog", "cat"], group_name="Normal"
    )
    # Cars and bicycles will go in the "Anomalous" group
    annotation_reader.group_labels(
        labels_to_group=["car", "bicycle"], group_name="Anomalous"
    )

    # Create project and upload data
    client.create_single_task_project_from_dataset(
        project_name=PROJECT_NAME,
        project_type=PROJECT_TYPE,
        path_to_images=PATH_TO_IMAGES_IN_COCO_DATASET,
        annotation_reader=annotation_reader,
        labels=LABELS_OF_INTEREST,
        number_of_images_to_upload=NUMBER_OF_IMAGES_TO_UPLOAD,
        number_of_images_to_annotate=NUMBER_OF_IMAGES_TO_ANNOTATE,
        enable_auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )

    print(
        "\n ------- Creating detection -> segmentation project --------------- \n"
    )

    LABELS_OF_INTEREST = ["dog", "cat", "horse"]
    PROJECT_TYPE = "detection_to_segmentation"
    PROJECT_NAME = "Animal detection to segmentation demo"

    label_source_per_task = []
    for task_type in get_task_types_by_project_type(PROJECT_TYPE):
        annotation_reader = DatumAnnotationReader(
            base_data_folder=PATH_TO_COCO_DATASET,
            annotation_format='coco',
            task_type=task_type
        )
        annotation_reader.filter_dataset(labels=LABELS_OF_INTEREST, criterion='OR')
        label_source_per_task.append(annotation_reader)

    # Group the labels for the first annotation reader, so that the detection task
    # will only see a single label
    label_source_per_task[0].group_labels(
        labels_to_group=["dog", "cat", "horse"], group_name="animal"
    )

    # Create project and upload data
    client.create_task_chain_project_from_dataset(
        project_name=PROJECT_NAME,
        project_type=PROJECT_TYPE,
        path_to_images=PATH_TO_IMAGES_IN_COCO_DATASET,
        label_source_per_task=label_source_per_task,
        number_of_images_to_upload=NUMBER_OF_IMAGES_TO_UPLOAD,
        number_of_images_to_annotate=NUMBER_OF_IMAGES_TO_ANNOTATE,
        enable_auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )

    print(
        "\n ------- Creating detection -> classification project --------------- \n"
    )

    LABELS_OF_INTEREST = ["dog", "cat", "horse"]
    PROJECT_TYPE = "detection_to_classification"
    PROJECT_NAME = "Animal detection to classification demo"

    label_source_per_task = []
    for task_type in get_task_types_by_project_type(PROJECT_TYPE):
        annotation_reader = DatumAnnotationReader(
            base_data_folder=PATH_TO_COCO_DATASET,
            annotation_format='coco',
            task_type=task_type
        )
        annotation_reader.filter_dataset(labels=LABELS_OF_INTEREST, criterion='OR')
        label_source_per_task.append(annotation_reader)

    # Group the labels for the first annotation reader, so that the detection task
    # will only see a single label
    label_source_per_task[0].group_labels(
        labels_to_group=["dog", "cat", "horse"], group_name="animal"
    )

    # Create project and upload data
    client.create_task_chain_project_from_dataset(
        project_name=PROJECT_NAME,
        project_type=PROJECT_TYPE,
        path_to_images=PATH_TO_IMAGES_IN_COCO_DATASET,
        label_source_per_task=label_source_per_task,
        number_of_images_to_upload=NUMBER_OF_IMAGES_TO_UPLOAD,
        number_of_images_to_annotate=NUMBER_OF_IMAGES_TO_ANNOTATE,
        enable_auto_train=AUTO_TRAIN_AFTER_UPLOAD
    )
