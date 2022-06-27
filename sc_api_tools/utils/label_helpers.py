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

from typing import List, Dict


def generate_segmentation_labels(detection_labels: List[str]) -> List[str]:
    """
    Generate segmentation label names from a list of detection label names.

    :param detection_labels: label names to generate segmentation label names for
    :return: List of label names
    """
    return [f"{label} shape" for label in detection_labels]


def generate_classification_labels(
        labels: List[str], multilabel: bool = False
) -> List[Dict[str, str]]:
    """
    Generates label creation data from a list of label names. If `multiclass = True`,
    the labels will be generated in such a way that multiple labels can be assigned to
    a single image. If `multiclass = False`, only a single label per image can be
    assigned.

    :param labels: Label names to be used
    :param multilabel: True to be able to assign multiple labels to one image, False
        otherwise. Defaults to False.
    :return: List of dictionaries containing the label data that can be sent to the SC
        project creation endpoint
    """
    label_list: List[Dict[str, str]] = []
    if multilabel or len(labels) == 1:
        for label in labels:
            label_list.append({"name": label, "group": f"{label}_group"})
    else:
        for label in labels:
            label_list.append({"name": label, "group": "default_classification_group"})
    return label_list
