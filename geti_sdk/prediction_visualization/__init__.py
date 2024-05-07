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

"""
Introduction
------------

The `prediction_visualization` package provides classes for visualizing models predictions.
Currently, the user interfaces to this package are available in the :py:mod:`~geti_sdk.utils.plot_helpers` module.

Module contents
---------------

.. automodule:: geti_sdk.prediction_visualization.visualizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: geti_sdk.prediction_visualization.shape_drawer.ShapeDrawer
   :members:
   :undoc-members:
   :show-inheritance:
"""

from .visualizer import Visualizer

__all__ = ["Visualizer"]
