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

The `rest_converters` package contains methods for serializing and deserializing
entities, using the data models defined in the :py:mod:`~geti_sdk.data_models`
module.

The rest converters are used in the :py:mod:`~geti_sdk.rest_clients`.

Module contents
---------------

.. autoclass:: ProjectRESTConverter
   :members:
   :undoc-members:

.. autoclass:: MediaRESTConverter
   :members:

.. autoclass:: AnnotationRESTConverter
   :members:

.. autoclass:: PredictionRESTConverter
   :members:

.. autoclass:: ConfigurationRESTConverter
   :members:

.. autoclass:: ModelRESTConverter
   :members:

.. autoclass:: StatusRESTConverter
   :members:

.. autoclass:: JobRESTConverter
   :members:

.. autoclass:: TestResultRESTConverter
   :members:
"""
from .annotation_rest_converter import AnnotationRESTConverter
from .configuration_rest_converter import ConfigurationRESTConverter
from .job_rest_converter import JobRESTConverter
from .media_rest_converter import MediaRESTConverter
from .model_rest_converter import ModelRESTConverter
from .prediction_rest_converter import PredictionRESTConverter
from .project_rest_converter import ProjectRESTConverter
from .status_rest_converter import StatusRESTConverter
from .test_result_rest_converter import TestResultRESTConverter

__all__ = [
    "ProjectRESTConverter",
    "MediaRESTConverter",
    "AnnotationRESTConverter",
    "PredictionRESTConverter",
    "ConfigurationRESTConverter",
    "ModelRESTConverter",
    "StatusRESTConverter",
    "JobRESTConverter",
    "TestResultRESTConverter",
]
