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

import logging
import time
from multiprocessing import Value
from queue import PriorityQueue
from threading import Event, Thread
from typing import Optional

import cv2
import numpy as np


def _process(
    queue: PriorityQueue,
    writer: cv2.VideoWriter,
    stop_event: Event,
    fps_limit: Optional[float] = None,
    buffer_size: int = 0,
):
    """
    Process video frames from a `queue`, by writing them to `writer`.
    If `fps_limit` is given, the rate at which frames are written is limited to at
    most the specified frequency.

    :param queue: PriorityQueue holding the video frames
    :param writer: VideoWriter object used to write the frames
    :param stop_event: Event object that signals the processing to stop when set
    :param fps_limit: Optional rate limit for writing the frames
    """
    if fps_limit is not None:
        next_frame = 0
        interval = 1 / fps_limit

    stop_received = False
    while True:
        if stop_event.is_set():
            if not stop_received:
                logging.info("Stop event received, waiting for queue to be emptied.")
                stop_received = False
            if queue.empty():
                logging.info("Video writing process stopped successfully.")
                writer.release()
                return
        if queue.qsize() < buffer_size and not stop_event.is_set():
            continue
        try:
            timestamp, index, frame = queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
        writer.write(frame)

        if fps_limit is not None:
            t = time.monotonic()
            if t < next_frame:
                time.sleep(next_frame - t)
                t = time.monotonic()
            next_frame = t + interval


class AsyncVideoProcessor:
    """
    Processor to write video frames to file or stream asynchronously, using a single
    thread and priority queue to sort frames based on timestamp
    """

    def __init__(
        self,
        writer: cv2.VideoWriter,
        fps_limit: Optional[float] = None,
        frame_buffer_size: int = 0,
    ):
        """
        Process video frames asynchronously, and stream them in the correct order to a
        video file or video stream.

        :param writer: VideoWriter object to write the frames to
        :param fps_limit: Optional rate limit of the writing process. If set, frames
            will be written to the file at at most this frequency
        :param frame_buffer_size: Size (in number of frames) of the buffer to keep for
            writing. If set too small, proper ordering of the frames is no longer
            guaranteed.
        """
        self.writer = writer
        self._queue = PriorityQueue()
        self.fps_limit = fps_limit
        self._stop_event = Event()
        self.buffer_size = frame_buffer_size
        self._worker = Thread(
            target=_process,
            kwargs={
                "queue": self._queue,
                "writer": self.writer,
                "stop_event": self._stop_event,
                "fps_limit": self.fps_limit,
                "buffer_size": self.buffer_size,
            },
        )
        self._counter = Value("i", 0)
        self._is_running: bool = False

    def enqueue(self, frame: np.ndarray, timestamp: float, index: Optional[int] = None):
        """
        Put a `frame` in the queue for processing.

        :param frame: Numpy array representing the video frame to put in the queue
        :param timestamp: Timestamp for the frame
        :param index: Optional index for the frame
        """
        if index is None:
            with self._counter.get_lock():
                index = self._counter.value
                self._counter.value += 1
        self._queue.put((timestamp, index, frame))

    def start(self):
        """
        Start the processing
        """
        logging.info("AsyncVideoProcessor started listening for frames")
        self._worker.start()
        self._is_running = True

    def stop(self):
        """
        Stop the processing
        """
        if not self._is_running:
            raise ValueError("Worker thread is not running, unable to stop processing")
        logging.info("Gracefully stopping video processing thread")
        self._stop_event.set()
        self._worker.join()

    def await_all(self):
        """
        Block execution until all frames have finished processing.
        Stops the worker thread once processing is done
        """
        if not (self._worker.is_alive() and self._is_running):
            raise RuntimeError("Worker thread is not running!")
        while not self._queue.empty():
            time.sleep(0.1)
        self.stop()
