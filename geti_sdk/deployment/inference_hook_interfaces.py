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
import logging
from abc import abstractmethod
from typing import Optional

import numpy as np

from geti_sdk.data_models import Prediction


class PostInferenceTrigger(object):
    """
    Base class for post inference triggers

    Post inference triggers are used in inference hooks. They define a condition
    (or set of conditions) that determines whether any post-inference action should be
    taken.

    :param threshold: Threshold to use in activating the trigger. If the computed
        score is above the threshold, the trigger emits a positive decision and any
        downstream actions will be performed
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._repr_info_ = f"threshold={threshold:.1f}"

    @abstractmethod
    def __call__(self, image: np.ndarray, prediction: Prediction) -> float:
        """
        Compute a trigger score for the `image` and corresponding `prediction`.

        :param image: Numpy array representing an image
        :param prediction: Prediction object corresponding to the inference result
            for the image.
        :return: Float representing the score for the input
        """
        return NotImplemented

    def get_decision(self, score: float) -> bool:
        """
        Make a decision based on a previously computed `score` and the threshold defined
        for the trigger

        :param score: Float representing the score for a certain image, prediction pair
        :return: Boolean indicating whether the trigger conditions are met (True,
            score is higher than the threshold), or not (False)
        """
        return score > self.threshold

    def __repr__(self) -> str:
        """
        Return string representation of the PostInferenceTrigger
        """
        return f"{type(self).__name__}({self._repr_info_})"


class PostInferenceAction(object):
    """
    Base class for post inference actions. These are actions that are used in inference
    hooks, and can be (conditionally) executed after inference

    :param log_level: Log level to use in the action, current options are `info` or
        `debug`. Defaults to `debug`
    """

    def __init__(self, log_level: str = "debug"):
        if log_level.lower() == "debug":
            self.log_function = logging.debug
        elif log_level.lower() == "info":
            self.log_function = logging.info
        else:
            raise ValueError(
                f"Unsupported log_level `{log_level}`, options are 'info' or 'debug'."
            )
        self._repr_info_: str = ""

    @abstractmethod
    def __call__(
        self, image: np.ndarray, prediction: Prediction, score: Optional[float] = None
    ):
        """
        Execute the action for the given `image` with corresponding `prediction` and an
        `score`. The `score` could be passed from the trigger that triggers the action.

        :param image: Numpy array representing an image
        :param prediction: Prediction object which was generated for the image
        :param score: Optional score computed from a post inference trigger
        """
        return NotImplemented

    def __repr__(self) -> str:
        """
        Return string representation of the PostInferenceAction
        """
        return f"{type(self).__name__}({self._repr_info_})"


class PostInferenceHookInterface(object):
    """
    Basic hook that can be executed after inference. A PostInferenceHook consists of:
      - A Trigger, defining a condition
      - An Action, defining an act or sequence of actions to perform when the
        trigger is activated.


    :param trigger: PostInferenceTrigger which is evaluated and used to trigger the
        hook action
    :param action: PostInferenceAction to be executed if triggered
    """

    def __init__(self, trigger: PostInferenceTrigger, action: PostInferenceAction):
        self.trigger = trigger
        self.action = action

    @abstractmethod
    def run(self, image: np.ndarray, prediction: Prediction) -> None:
        """
        Execute the post inference hook. This will first evaluate the `trigger`, and
        if the trigger conditions are met the corresponding `action` will be executed.

        :param image: Numpy array representing the image that was the input for
            model inference
        :param prediction: Prediction object containing the inference results for the
            image
        """
        raise NotImplementedError
