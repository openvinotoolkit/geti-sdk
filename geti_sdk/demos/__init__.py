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

"""
Introduction
------------

The `demos` package contains useful functions for setting up demo projects on any
Intel® Geti™ server.

Module contents
---------------

"""
from .constants import DEFAULT_DATA_PATH, NOTEBOOK_DATA_PATH
from .data_helpers import get_coco_dataset, get_mvtec_dataset
from .demo_projects import (
    create_anomaly_classification_demo_project,
    create_classification_demo_project,
    create_detection_demo_project,
    create_detection_to_classification_demo_project,
    create_detection_to_segmentation_demo_project,
    create_segmentation_demo_project,
    ensure_project_is_trained,
    ensure_trained_anomaly_project,
    ensure_trained_example_project,
)

__all__ = [
    "DEFAULT_DATA_PATH",
    "NOTEBOOK_DATA_PATH",
    "create_segmentation_demo_project",
    "create_anomaly_classification_demo_project",
    "create_classification_demo_project",
    "create_detection_demo_project",
    "create_detection_to_segmentation_demo_project",
    "create_detection_to_classification_demo_project",
    "ensure_trained_example_project",
    "ensure_trained_anomaly_project",
    "ensure_project_is_trained",
    "get_coco_dataset",
    "get_mvtec_dataset",
]
