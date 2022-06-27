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
entities, using the data models defined in the :py:mod:`~sc_api_tools.data_models`
module.

The rest converters are used in the :py:mod:`~sc_api_tools.rest_managers`.

Module contents
---------------

.. autoclass:: ProjectRESTConverter
   :members:
   :undoc-members:

.. autoclass:: MediaRESTConverter
   :members:

.. autoclass:: annotation_rest_converter.AnnotationRESTConverter
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

"""
from .project_rest_converter import ProjectRESTConverter
from .media_rest_converter import MediaRESTConverter
from .annotation_rest_converter import AnnotationRESTConverter
from .prediction_rest_converter import PredictionRESTConverter
from .configuration_rest_converter import ConfigurationRESTConverter
from .model_rest_converter import ModelRESTConverter
from .status_rest_converter import StatusRESTConverter
from .job_rest_converter import JobRESTConverter

__all__ = [
    "ProjectRESTConverter",
    "MediaRESTConverter",
    "AnnotationRESTConverter",
    "PredictionRESTConverter",
    "ConfigurationRESTConverter",
    "ModelRESTConverter",
    "StatusRESTConverter",
    "JobRESTConverter"
]
