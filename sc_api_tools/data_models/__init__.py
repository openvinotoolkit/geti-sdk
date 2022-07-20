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

The `data_models` package contains the SDK representation for all entities in SC, such
as :py:class:`~sc_api_tools.data_models.annotation_scene.AnnotationScene`,
:py:class:`~sc_api_tools.data_models.media.Image`,
:py:class:`~sc_api_tools.data_models.project.Project` and
:py:class:`~sc_api_tools.data_models.model.Model` and many more.

When interacting with the SC cluster through the
:py:class:`sc_api_tools.sc_rest_client.SCRESTClient` or the
:py:mod:`~sc_api_tools.rest_clients`, all entities retrieved from the cluster will be
deserialized into the data models defined in this package.

Module contents
---------------

Algorithm-related entities
++++++++++++++++++++++++++

.. automodule:: sc_api_tools.data_models.algorithms
   :members:
   :undoc-members:

Project-related entities
++++++++++++++++++++++++

.. automodule:: sc_api_tools.data_models.task
   :members:
   :undoc-members:

.. automodule:: sc_api_tools.data_models.project
   :members:
   :undoc-members:

Annotation-related entities
+++++++++++++++++++++++++++
.. automodule:: sc_api_tools.data_models.label
   :members:
   :undoc-members:

.. automodule:: sc_api_tools.data_models.shapes
   :members:
   :undoc-members:

.. automodule:: sc_api_tools.data_models.annotations
   :members:
   :undoc-members:

.. automodule:: sc_api_tools.data_models.annotation_scene
   :members:
   :undoc-members:

Configuration-related entities
++++++++++++++++++++++++++++++

.. automodule:: sc_api_tools.data_models.configurable_parameter
   :members:
   :undoc-members:

.. automodule:: sc_api_tools.data_models.configurable_parameter_group
   :members:
   :undoc-members:

.. automodule:: sc_api_tools.data_models.configuration_identifiers
   :members:
   :undoc-members:

.. automodule:: sc_api_tools.data_models.configuration
   :members:
   :undoc-members:

Model-related entities
++++++++++++++++++++++

.. automodule:: sc_api_tools.data_models.model
   :members:
   :undoc-members:

.. automodule:: sc_api_tools.data_models.model_group
   :members:
   :undoc-members:

Media-related entities
++++++++++++++++++++++

.. automodule:: sc_api_tools.data_models.media
   :members:
   :undoc-members:

.. automodule:: sc_api_tools.data_models.media_identifiers
   :members:
   :undoc-members:

.. automodule:: sc_api_tools.data_models.task_annotation_state
   :members:
   :undoc-members:

Prediction-related entities
+++++++++++++++++++++++++++

.. automodule:: sc_api_tools.data_models.predictions
   :members:
   :undoc-members:

Status- and job-related entities
++++++++++++++++++++++++++++++++

.. automodule:: sc_api_tools.data_models.status
   :members:
   :undoc-members:

.. automodule:: sc_api_tools.data_models.job
   :members:
   :undoc-members:

Utility functions
+++++++++++++++++

.. automodule:: sc_api_tools.data_models.utils
   :members:
   :undoc-members:

Custom container classes
++++++++++++++++++++++++

.. automodule:: sc_api_tools.data_models.containers
   :members:
   :undoc-members:

Enumerations
++++++++++++

.. automodule:: sc_api_tools.data_models.enums
   :members:
   :undoc-members:

"""

from .algorithms import Algorithm
from .annotation_scene import AnnotationScene
from .annotations import Annotation
from .configuration import (
    ConfigurableParameters,
    FullConfiguration,
    GlobalConfiguration,
    TaskConfiguration,
)
from .enums import AnnotationKind, MediaType, TaskType
from .job import Job
from .label import Label, ScoredLabel
from .media import Image, MediaItem, Video, VideoFrame
from .model import Model, OptimizedModel
from .model_group import ModelGroup, ModelSummary
from .performance import Performance
from .predictions import Prediction
from .project import Pipeline, Project
from .status import ProjectStatus
from .task import Task

__all__ = [
    "TaskType",
    "AnnotationKind",
    "Project",
    "Label",
    "Task",
    "Pipeline",
    "Image",
    "Video",
    "MediaItem",
    "MediaType",
    "Annotation",
    "AnnotationScene",
    "Algorithm",
    "ScoredLabel",
    "VideoFrame",
    "Prediction",
    "Performance",
    "TaskConfiguration",
    "GlobalConfiguration",
    "ConfigurableParameters",
    "FullConfiguration",
    "Model",
    "ModelGroup",
    "OptimizedModel",
    "ModelSummary",
    "ProjectStatus",
    "Job",
]
