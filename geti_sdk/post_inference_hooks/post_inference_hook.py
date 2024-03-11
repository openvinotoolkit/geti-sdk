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
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from geti_sdk.data_models import Prediction
from geti_sdk.deployment.inference_hook_interfaces import (
    PostInferenceAction,
    PostInferenceHookInterface,
    PostInferenceTrigger,
)
from geti_sdk.post_inference_hooks.utils import RateLimiter


class PostInferenceHook(PostInferenceHookInterface):
    """
    Basic hook that can be executed after inference. A PostInferenceHook consists of:
      - A Trigger, defining a condition
      - An Action, defining an act or sequence of actions to perform when the
        trigger is activated.

    By default, hooks are executed in parallel in a separate thread from the main
    inference code. If you want to disable parallel execution, make sure to set
    `max_threads=0` when creating the hook.

    :param trigger: PostInferenceTrigger which is evaluated and used to trigger the
        hook action
    :param action: PostInferenceAction to be executed if triggered
    :param max_threads: Maximum number of threads to use for hook execution. Defaults
        to 5. Set to 0 to disable parallel hook execution.
    :param limit_action_rate: True to limit the rate at which the `action` for the
        hook is executed. Note that rate limiting is applied **after** trigger
        evaluation, so this can be useful for handling video streaming in which
        subsequent frames may be very similar. Enabling rate limiting will avoid
        sampling many near-duplicate frames
    :param max_frames_per_second: Maximum frame rate at which to execute the action.
        Only takes effect if `limit_action_rate = True`.
    """

    def __init__(
        self,
        trigger: PostInferenceTrigger,
        action: PostInferenceAction,
        max_threads: int = 5,
        limit_action_rate: bool = False,
        max_frames_per_second: float = 1,
    ):
        self.trigger = trigger
        self.action = action
        self.parallel_execution = max_threads != 0
        if self.parallel_execution:
            self.executor = ThreadPoolExecutor(max_workers=max_threads)

        if limit_action_rate:
            self.rate_limiter: Optional[RateLimiter] = RateLimiter(
                frames_per_second=max_frames_per_second, is_blocking=False
            )
        else:
            self.rate_limiter = None

    def run(self, image: np.ndarray, prediction: Prediction) -> None:
        """
        Run the post inference hook. First evaluate the trigger, then execute the
        action if the trigger conditions are met.

        :param image: Numpy array representing the image
        :param prediction: Prediction object containing the inference results for the
            image
        """

        def execution_function(im: np.ndarray, pred: Prediction) -> None:
            score = self.trigger(image=im, prediction=pred)
            decision = self.trigger.get_decision(score=score)
            if decision:
                if self.rate_limiter is not None:
                    take_action = next(self.rate_limiter)
                else:
                    take_action = True
                if take_action:
                    self.action(image=im, prediction=pred, score=score)

        if self.parallel_execution:
            self.executor.submit(execution_function, image, prediction)
        else:
            execution_function(image, prediction)

    def __del__(self):
        """
        Await shutdown of the ThreadPoolExecutor, if needed.

        Called upon deletion of the post inference hook or garbage collection
        """
        if self.parallel_execution:
            self.executor.shutdown(wait=True)
