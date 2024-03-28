# Copyright (C) 2024 Intel Corporation
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
import copy
from typing import List

import numpy as np

from geti_sdk.data_models import Prediction
from geti_sdk.deployment.inference_hook_interfaces import PostInferenceTrigger


class LabelTrigger(PostInferenceTrigger):
    """
    Post inference trigger based on the presence of certain label(s) in the
    predictions for an image.
    If the specified label or labels are in the prediction results for an inferred
    image, the trigger is activated.

    :param label_names: List of label names to search for. If any of
    :param mode: Operation mode to use when evaluating the labels. Supported modes are:
        - `OR` (the default) -> If **any** of the labels in `label_names` if found in
            the prediction, the trigger activates.
        - `AND` -> If and only if **all** of the labels in `label_names` are
            predicted within the image, the trigger activates
    """

    def __init__(self, label_names: List[str], mode: str = "OR"):
        self.label_names = label_names
        self.mode = mode

        # LabelTrigger will return a score of 1 if label is found, so we can use the
        # default threshold defined in the super class
        super().__init__()
        if label_names is not None:
            self._repr_info_ += f", label_names={label_names}"
        self._repr_info_ += f", mode={mode}"

    def __call__(self, image: np.ndarray, prediction: Prediction) -> float:
        """
        Compute a trigger score for the `image` and corresponding `prediction`.

        :param image: Numpy array representing an image
        :param prediction: Prediction object corresponding to the inference result
            for the image.
        :return: Float representing the score for the input
        """
        if self.mode == "AND":
            labels_to_search = copy.deepcopy(self.label_names)
        else:
            labels_to_search = self.label_names
        for predicted_object in prediction.annotations:
            for label in predicted_object.labels:
                if label.name in labels_to_search:
                    if self.mode == "OR":
                        return 1
                    elif self.mode == "AND":
                        labels_to_search.remove(label.name)
                        if len(labels_to_search) == 0:
                            return 1
        return 0
