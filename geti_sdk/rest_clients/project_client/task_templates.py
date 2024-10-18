# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

BASE_TEMPLATE = {
    "name": "dummy project name",
    "pipeline": {
        "connections": [],
        "tasks": [{"title": "Dataset", "task_type": "dataset"}],
    },
}

DETECTION_TASK = {"title": "Detection task", "task_type": "detection", "labels": []}

ROTATED_DETECTION_TASK = {
    "title": "Rotated detection task",
    "task_type": "rotated_detection",
    "labels": [],
}

SEGMENTATION_TASK = {
    "title": "Segmentation task",
    "task_type": "segmentation",
    "labels": [],
}

INSTANCE_SEGMENTATION_TASK = {
    "title": "Instance segmentation task",
    "task_type": "instance_segmentation",
    "labels": [],
}

CLASSIFICATION_TASK = {
    "title": "Classification task",
    "task_type": "classification",
    "labels": [],
}

ANOMALY_CLASSIFICATION_TASK = {
    "title": "Anomaly classification task",
    "task_type": "anomaly_classification",
    "labels": [],
}

# This is the reduced anomaly task.
# It goes under `Anomaly` title,
# and it is `Anomally classification task` under the hood
ANOMALY_TASK = {
    "title": "Anomaly",
    "task_type": "anomaly",
    "labels": [],
}

ANOMALY_DETECTION_TASK = {
    "title": "Anomaly detection task",
    "task_type": "anomaly_detection",
    "labels": [],
}

ANOMALY_SEGMENTATION_TASK = {
    "title": "Anomaly segmentation task",
    "task_type": "anomaly_segmentation",
    "labels": [],
}

INSTANCE_SEGMENTATION_TASK = {
    "title": "Instance segmentation task",
    "task_type": "instance_segmentation",
    "labels": [],
}

ROTATED_DETECTION_TASK = {
    "title": "Rotated detection task",
    "task_type": "rotated_detection",
    "labels": [],
}

CROP_TASK = {"title": "Crop task", "task_type": "crop"}
