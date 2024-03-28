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
import numpy as np

from geti_sdk.data_models import Prediction
from geti_sdk.deployment.inference_hook_interfaces import PostInferenceTrigger


class AlwaysTrigger(PostInferenceTrigger):
    """
    Post inference trigger that activates on each inference call.
    """

    def __init__(self):
        super().__init__()
        self._repr_info_ = ""

    def __call__(self, image: np.ndarray, prediction: Prediction) -> float:
        """
        Compute a trigger score for the `image` and corresponding `prediction`.

        :param image: Numpy array representing an image
        :param prediction: Prediction object corresponding to the inference result
            for the image.
        :return: Float representing the score for the input
        """
        return 1
