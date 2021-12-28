import os
from typing import Optional

from sc_api_tools import SCRESTClient
from sc_api_tools.annotation_readers import DatumAnnotationReader
from sc_api_tools.data_models import Project
from sc_api_tools.utils import get_coco_dataset, get_task_types_by_project_type
from sc_api_tools.utils.data_download_helpers import (
    COCOSubset,
    directory_has_coco_subset
)

DEFAULT_COCO_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    ), 'data'
)


def is_coco_dataset(dataset_path: Optional[str] = None):
    """
    This method checks if the COCO dataset is present at the specified path. If not,
    this method will attempt to download the dataset to the path specified.

    :param dataset_path: Path to check against.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_COCO_PATH
    get_coco_dataset(dataset_path, verbose=True)


def create_segmentation_demo_project(
        client: SCRESTClient,
        n_images: int,
        n_annotations: int = -1,
        auto_train: bool = False,
        dataset_path: Optional[str] = None
) -> Project:
    """
    This method creates a demo project of type 'segmentation', based off the MS COCO
    dataset.

    It creates a project with a single 'Segmentation' task, with segmentation mask for
    labels: 'dog' and 'frisbee'.

    :param client: SCRESTClient, representing the client for the SC cluster on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param auto_train: True to set auto-training to True once the project has been
        created and the images have been annotated, False to leave auto-training
        turned off.
    :param dataset_path: Path to the COCO dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the sc_api_tools package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :return: Project object, holding detailed information about the project that was
        created on the SC cluster.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_COCO_PATH
    coco_path = get_coco_dataset(dataset_path)
    print("\n ------- Creating segmentation project --------------- \n")

    labels_of_interest = ["dog", "frisbee"]
    project_type = "segmentation"
    project_name = "Segmentation demo"

    # Create annotation reader
    annotation_reader = DatumAnnotationReader(
        base_data_folder=coco_path,
        annotation_format='coco'
    )
    annotation_reader.filter_dataset(
        labels=labels_of_interest, criterion='OR'
    )
    # Create project and upload data
    return client.create_single_task_project_from_dataset(
        project_name=project_name,
        project_type=project_type,
        path_to_images=coco_path,
        annotation_reader=annotation_reader,
        labels=labels_of_interest,
        number_of_images_to_upload=n_images,
        number_of_images_to_annotate=n_annotations,
        enable_auto_train=auto_train
    )


def create_detection_demo_project(
        client: SCRESTClient,
        n_images: int,
        n_annotations: int = -1,
        auto_train: bool = False,
        dataset_path: Optional[str] = None
) -> Project:
    """
    This method creates a demo project of type 'detection', based off the MS COCO
    dataset.

    It creates a project with a single 'Detection' task, with labels and annotations
    for 'cell phone' and 'person' objects

    :param client: SCRESTClient, representing the client for the SC cluster on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param auto_train: True to set auto-training to True once the project has been
        created and the images have been annotated, False to leave auto-training
        turned off.
    :param dataset_path: Path to the COCO dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the sc_api_tools package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :return: Project object, holding detailed information about the project that was
        created on the SC cluster.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_COCO_PATH
    coco_path = get_coco_dataset(dataset_path)
    print("\n ------- Creating detection project --------------- \n")

    labels_of_interest = ["cell phone", "person"]
    project_type = "detection"
    project_name = "Detection demo"

    # Create annotation reader
    annotation_reader = DatumAnnotationReader(
        base_data_folder=coco_path,
        annotation_format='coco'
    )
    annotation_reader.filter_dataset(
        labels=labels_of_interest, criterion='AND'
    )
    # Create project and upload data
    return client.create_single_task_project_from_dataset(
        project_name=project_name,
        project_type=project_type,
        path_to_images=coco_path,
        annotation_reader=annotation_reader,
        labels=labels_of_interest,
        number_of_images_to_upload=n_images,
        number_of_images_to_annotate=n_annotations,
        enable_auto_train=auto_train
    )


def create_classification_demo_project(
        client: SCRESTClient,
        n_images: int,
        n_annotations: int = -1,
        auto_train: bool = False,
        dataset_path: Optional[str] = None
) -> Project:
    """
    This method creates a demo project of type 'classification', based off the MS COCO
    dataset

    It creates a project with a single 'Classification' task, with labels to classify
    images with different types of animals. The following animals are considered:
    "horse", "cat", "zebra", "bear"

    :param client: SCRESTClient, representing the client for the SC cluster on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param auto_train: True to set auto-training to True once the project has been
        created and the images have been annotated, False to leave auto-training
        turned off.
    :param dataset_path: Path to the COCO dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the sc_api_tools package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :return: Project object, holding detailed information about the project that was
        created on the SC cluster.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_COCO_PATH
    coco_path = get_coco_dataset(dataset_path)
    print("\n ------- Creating classification project --------------- \n")

    labels_of_interest = ["horse", "cat", "zebra", "bear"]
    project_type = "classification"
    project_name = "Classification demo"

    # Create annotation reader
    annotation_reader = DatumAnnotationReader(
        base_data_folder=coco_path,
        annotation_format='coco'
    )
    annotation_reader.filter_dataset(
        labels=labels_of_interest, criterion='XOR'
    )
    # Create project and upload data
    return client.create_single_task_project_from_dataset(
        project_name=project_name,
        project_type=project_type,
        path_to_images=coco_path,
        annotation_reader=annotation_reader,
        labels=labels_of_interest,
        number_of_images_to_upload=n_images,
        number_of_images_to_annotate=n_annotations,
        enable_auto_train=auto_train
    )


def create_anomaly_classification_demo_project(
        client: SCRESTClient,
        n_images: int,
        n_annotations: int = -1,
        auto_train: bool = False,
        dataset_path: Optional[str] = None
) -> Project:
    """
    This method creates a demo project of type 'anomaly_classification', based off the
    MS COCO dataset.

    It creates a project with a single 'Anomaly classification' task. Images with
    animals in them are considered 'Normal', whereas images with traffic lights or
    stop signs in them are considered 'Anomalous'

    :param client: SCRESTClient, representing the client for the SC cluster on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param auto_train: True to set auto-training to True once the project has been
        created and the images have been annotated, False to leave auto-training
        turned off.
    :param dataset_path: Path to the COCO dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the sc_api_tools package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :return: Project object, holding detailed information about the project that was
        created on the SC cluster.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_COCO_PATH
    coco_path = get_coco_dataset(dataset_path)
    print("\n ------- Creating anomaly classification project --------------- \n")

    animal_labels = ["horse", "dog", "cat", "elephant", "giraffe", "cow", "sheep"]
    traffic_labels = ["stop sign", "traffic light"]

    labels_of_interest = animal_labels + traffic_labels
    project_type = "anomaly_classification"
    project_name = "Anomaly classification demo"

    # Create annotation reader
    annotation_reader = DatumAnnotationReader(
        base_data_folder=coco_path,
        annotation_format='coco'
    )
    annotation_reader.filter_dataset(
        labels=labels_of_interest, criterion='XOR'
    )

    # Create groups of "Normal" and "Anomalous" images, since those are the only
    # label names allowed for anomaly classification
    # All animals will be considered "Normal"
    annotation_reader.group_labels(
        labels_to_group=animal_labels, group_name="Normal"
    )
    # Cars and bicycles will go in the "Anomalous" group
    annotation_reader.group_labels(
        labels_to_group=traffic_labels, group_name="Anomalous"
    )

    # Create project and upload data
    return client.create_single_task_project_from_dataset(
        project_name=project_name,
        project_type=project_type,
        path_to_images=coco_path,
        annotation_reader=annotation_reader,
        labels=labels_of_interest,
        number_of_images_to_upload=n_images,
        number_of_images_to_annotate=n_annotations,
        enable_auto_train=auto_train
    )


def create_detection_to_segmentation_demo_project(
        client: SCRESTClient,
        n_images: int,
        n_annotations: int = -1,
        auto_train: bool = False,
        dataset_path: Optional[str] = None
) -> Project:
    """
    This method creates a demo project of type 'detection_to_segmentation', based
    off the MS COCO dataset.

    It creates a project with a 'Detection' task, followed by a 'Segmentation' task.
    The detection task has the label 'animal', and the segmentation task has
    annotations for the following species:
        "dog", "cat", "horse", "elephant", "cow", "sheep", "giraffe", "zebra", "bear"


    :param client: SCRESTClient, representing the client for the SC cluster on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param auto_train: True to set auto-training to True once the project has been
        created and the images have been annotated, False to leave auto-training
        turned off.
    :param dataset_path: Path to the COCO dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the sc_api_tools package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :return: Project object, holding detailed information about the project that was
        created on the SC cluster.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_COCO_PATH
    coco_path = get_coco_dataset(dataset_path)
    print(
        "\n ------- Creating detection -> segmentation project --------------- \n"
    )
    animal_labels = [
        "dog", "cat", "horse", "cow", "sheep"
    ]
    project_type = "detection_to_segmentation"
    project_name = "Animal detection to segmentation demo"

    label_source_per_task = []
    for task_type in get_task_types_by_project_type(project_type):
        annotation_reader = DatumAnnotationReader(
            base_data_folder=coco_path,
            annotation_format='coco',
            task_type=task_type
        )
        annotation_reader.filter_dataset(labels=animal_labels, criterion='OR')
        label_source_per_task.append(annotation_reader)

    # Group the labels for the first annotation reader, so that the detection task
    # will only see a single label
    label_source_per_task[0].group_labels(
        labels_to_group=animal_labels, group_name="animal"
    )

    # Create project and upload data
    return client.create_task_chain_project_from_dataset(
        project_name=project_name,
        project_type=project_type,
        path_to_images=coco_path,
        label_source_per_task=label_source_per_task,
        number_of_images_to_upload=n_images,
        number_of_images_to_annotate=n_annotations,
        enable_auto_train=auto_train
    )


def create_detection_to_classification_demo_project(
        client: SCRESTClient,
        n_images: int,
        n_annotations: int = -1,
        auto_train: bool = False,
        dataset_path: Optional[str] = None
) -> Project:
    """
    This method creates a demo project of type 'detection_to_classification', based
    off the MS COCO dataset.

    It creates a project with a 'Detection' task, followed by a 'Classification' task.
    The detection task has the label 'animal', and the detection task discriminates
    between 'domestic' and 'wild' animals

    :param client: SCRESTClient, representing the client for the SC cluster on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param auto_train: True to set auto-training to True once the project has been
        created and the images have been annotated, False to leave auto-training
        turned off.
    :param dataset_path: Path to the COCO dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the sc_api_tools package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :return: Project object, holding detailed information about the project that was
        created on the SC cluster.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_COCO_PATH
    coco_path = get_coco_dataset(dataset_path)
    print(
        "\n ------- Creating detection -> segmentation project --------------- \n"
    )
    domestic_labels = ["dog", "cat", "horse", "cow", "sheep"]
    wild_labels = ["elephant", "giraffe", "zebra", "bear"]
    animal_labels = domestic_labels + wild_labels

    project_type = "detection_to_classification"
    project_name = "Animal detection to classification demo"

    label_source_per_task = []
    for task_type in get_task_types_by_project_type(project_type):
        annotation_reader = DatumAnnotationReader(
            base_data_folder=coco_path,
            annotation_format='coco',
            task_type=task_type
        )
        annotation_reader.filter_dataset(labels=animal_labels, criterion='OR')
        label_source_per_task.append(annotation_reader)

    # Group the labels for the first annotation reader, so that the detection task
    # will only see a single label
    label_source_per_task[0].group_labels(
        labels_to_group=animal_labels, group_name="animal"
    )
    # Group the labels for the second annotation reader
    label_source_per_task[1].group_labels(
        labels_to_group=domestic_labels, group_name='domestic'
    )
    label_source_per_task[1].group_labels(
        labels_to_group=wild_labels, group_name="wild"
    )
    # Create project and upload data
    return client.create_task_chain_project_from_dataset(
        project_name=project_name,
        project_type=project_type,
        path_to_images=coco_path,
        label_source_per_task=label_source_per_task,
        number_of_images_to_upload=n_images,
        number_of_images_to_annotate=n_annotations,
        enable_auto_train=auto_train
    )
