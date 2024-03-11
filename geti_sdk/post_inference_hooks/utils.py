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
"""Utility functions for the post inference hook module"""
import time
from collections import Iterator
from threading import Lock


class RateLimiter(Iterator):
    """
    Iterator that yields at a maximum frequency of `frames_per_second`.

    :param frames_per_second: Maximum execution rate, defined in frames per second
    :param is_blocking: Setting this to True will block execution until the next
        yield. Setting this to False will cause the Iterator to yield `False` if it
        is called before the next yield is available, and True otherwise
    """

    def __init__(self, frames_per_second: float = 1, is_blocking: bool = False):
        self.lock = Lock()
        self.interval = 1 / frames_per_second
        self.next_yield = 0
        self.is_blocking = is_blocking

    def __next__(self) -> bool:
        """
        Yield `True` if called after the `self.interval` has elapsed with respect to
        the last call. If `self.is_blocking == True`, will block execution otherwise,
        until the interval has elapsed. If `self.is_blocking == False`, execution is
        not blocked but `False` is yielded instead.
        """
        with self.lock:
            t = time.monotonic()
            if t < self.next_yield:
                if self.is_blocking:
                    time.sleep(self.next_yield - t)
                    t = time.monotonic()
                else:
                    yield False
            self.next_yield = t + self.interval
            yield True
