from .coco_demos import (
    create_detection_demo_project,
    create_classification_demo_project,
    create_segmentation_demo_project,
    create_anomaly_classification_demo_project,
    create_detection_to_segmentation_demo_project,
    create_detection_to_classification_demo_project,
    is_coco_dataset
)

__all__ = [
    "create_detection_demo_project",
    "create_classification_demo_project",
    "create_segmentation_demo_project",
    "create_anomaly_classification_demo_project",
    "create_detection_to_segmentation_demo_project",
    "create_detection_to_classification_demo_project",
    "is_coco_dataset"
]
