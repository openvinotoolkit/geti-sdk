BASE_TEMPLATE = {
    "name": "dummy project name",
    "pipeline": {
        "connections": [],
        "tasks": [
            {
                "title": "Dataset",
                "task_type": "dataset"
            }
        ]
    }

}

DETECTION_TASK = {
    "title": "Detection task",
    "task_type": "detection",
    "labels": []
}

SEGMENTATION_TASK = {
    "title": "Segmentation task",
    "task_type": "segmentation",
    "labels": []
}

CLASSIFICATION_TASK = {
    "title": "Classification task",
    "task_type": "classification",
    "labels": []
}

ANOMALY_CLASSIFICATION_TASK = {
    "title": "Anomaly classification task",
    "task_type": "anomaly_classification",
    "labels": []
}

CROP_TASK = {
    "title": "Crop task",
    "task_type": "crop"
}
