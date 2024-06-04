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
import os
import time
from multiprocessing import Value
from queue import Empty
from threading import Thread
from typing import Any, Callable, Optional, Tuple

import numpy as np

from geti_sdk.data_models import Prediction
from geti_sdk.deployment import Deployment

from .ordered_buffer import OrderedResultBuffer


class AsyncVideoProcessor:
    """
    A helper class for efficient and optimized frame-by-frame inference on videos
    """

    def __init__(
        self,
        deployment: Deployment,
        processing_function: Callable[[np.ndarray, Prediction, Any], None],
        min_buffer_size: Optional[int] = None,
        max_buffer_size: Optional[int] = None,
    ):
        """
        Run efficient and optimized frame-by-frame inference on videos. The
        AsyncVideoProcessor runs inference asynchronously,
        maximizing the host systems potential for parallelization. However, the
        internal buffer ensures that the video frame and inference result
        postprocessing is done in the original frame order.

        :param deployment: Model deployment to use for frame inference
        :param processing_function: A user defined function for further processing
            the video frames and the predictions. The function should take the
            following arguments:
                - image: np.ndarry containing the original image/video frame
                - prediction: Prediction object containing the model predictions for
                    the image/video frame
                - runtime_data: Any additional data that is required for the further
                    processing. Can be for example a filename, frame index, etc.
        :param min_buffer_size: Minimal number of frames to keep in the internal buffer
        :param max_buffer_size: Maximum number of frames that the buffer can hold.
            This can be used to limit the memory usage of the VideoProcessor
        """
        self.deployment = deployment
        self.processing_function = processing_function
        if min_buffer_size is None and os.cpu_count() is not None:
            min_buffer_size = 2 * os.cpu_count()
            logging.debug(
                f"Minimum buffer size set to `{min_buffer_size}`, twice the "
                f"number of available cpu's on the system."
            )
        elif min_buffer_size is not None and os.cpu_count() is not None:
            if min_buffer_size < os.cpu_count():
                logging.warning(
                    "The minimum buffer size is less than the cpu count of the system. "
                    "This may result in video frames not being processed in order. It"
                    "is recommended to increase the minimum buffer size."
                )
        self.buffer = OrderedResultBuffer(
            maxsize=max_buffer_size, minsize=min_buffer_size
        )
        self._worker: Optional[Thread] = None
        self._should_stop: bool = False
        self._should_stop_now: bool = False
        self._is_running: bool = False
        self._min_buffer_size = min_buffer_size
        self._current_index = Value("i", 0)
        self._processed_count = Value("i", 0)

        if deployment.asynchronous_mode:
            logging.info(
                "Deployment is already in asynchronous mode, any previously defined "
                "callback will be overwritten by the AsyncVideoProcessor"
            )

        if not deployment.are_models_loaded:
            logging.info(
                "Inference models are not loaded, configuring them for maximal "
                "throughput and loading to CPU now."
            )
            deployment.load_inference_models(
                device="CPU",
                max_async_infer_requests=os.cpu_count(),
                openvino_configuration={"PERFORMANCE_HINT": "THROUGHPUT"},
            )

        def infer_callback(
            image: np.ndarray, prediction: Prediction, runtime_data: Tuple[int, Any]
        ):
            """
            Infer callback to put the image, prediction and runtime data into the
            ordered buffer.
            """
            index, runtime_data = runtime_data
            self.buffer.put(
                index=index,
                image=image,
                prediction=prediction,
                runtime_data=runtime_data,
            )

        self.deployment.set_asynchronous_callback(infer_callback)

    def start(self, num_frames: Optional[int] = None):
        """
        Start the processing thread.

        :param num_frames: Optional integer specifying the length of the video to
            process. When processing a continuous stream, leave this as None
        """
        if self._is_running:
            raise ValueError("The processing thread is already running.")

        # Reset the index
        with self._current_index.get_lock():
            self._current_index.value = 0

        # Reset the processing count
        with self._processed_count.get_lock():
            self._processed_count.value = 0

        def process_items():
            """
            Process an item from the buffer
            """
            while True:
                if self._should_stop_now:
                    logging.debug("Stopping processing thread immediately")
                    return

                # Check if all frames have been inferred and put to the buffer already
                all_frames_inferred = False
                if num_frames is not None:
                    all_frames_inferred = (
                        self.buffer.total_items_buffered == num_frames - 1
                    )

                # In case of the `should_stop` signal, check if the buffer has reached
                # it's minimum size yet. If so, we can start to empty it
                processed_up_to_min_buffer = False
                if self._should_stop:
                    with (
                        self._current_index.get_lock(),
                        self._processed_count.get_lock(),
                    ):
                        if (
                            self._current_index.value - self._processed_count.value
                            <= self._min_buffer_size
                        ):
                            processed_up_to_min_buffer = True
                empty_buffer = processed_up_to_min_buffer or all_frames_inferred

                # Get the item from the buffer
                try:
                    item = self.buffer.get(empty_buffer=empty_buffer)
                except Empty:
                    with (
                        self._processed_count.get_lock(),
                        self._current_index.get_lock(),
                    ):
                        if self._processed_count.value == self._current_index.value:
                            # Buffer is empty, all items have been processed.
                            # Stop thread
                            return
                    time.sleep(1e-12)
                    continue

                self.processing_function(item.image, item.prediction, item.runtime_data)
                with self._processed_count.get_lock():
                    self._processed_count.value += 1

        self._worker = Thread(target=process_items, daemon=False)
        self._worker.start()
        self._is_running = True
        logging.debug("AsyncVideoProcessor: Processing thread started")

    def stop(self):
        """
        Stop the processing thread immediately. Any items left in the buffer will not
        be processed.
        """
        self._should_stop_now = True
        self._worker.join()
        self._is_running = False
        self._should_stop_now = False
        logging.debug("AsyncVideoProcessor: Processing thread stopped")

    def process(self, frame: np.ndarray, runtime_data: Optional[Any] = None):
        """
        Run inference for the frame and process the results according to the
        `processing_function` defined in the AsyncVideoProcessor initialization.

        :param frame: Numpy array representing the frame, in RGB-channel order
        :param runtime_data: Additional optional data pertaining to the frame, which
            should be passed to the processing_function
        """
        if not self._is_running:
            raise ValueError(
                "The processing thread is not running, please start it using the "
                "`.start()` method first."
            )
        with self._current_index.get_lock():
            self.deployment.infer_async(
                image=frame, runtime_data=(self._current_index.value, runtime_data)
            )
            self._current_index.value += 1

    def await_all(self):
        """
        Block program execution until all frames left in the buffer have been fully
        processed.
        """
        if not self._is_running:
            return
        self._should_stop = True
        self._worker.join()
        self._is_running = False
        self._should_stop = False
        while True:
            if self.buffer.is_empty and not self._worker.is_alive():
                # Buffer should be empty and worker thread should have stopped
                break
            time.sleep(1e-9)
        logging.debug(
            "AsyncVideoProcessor: All frames processed. Processing thread stopped"
        )
