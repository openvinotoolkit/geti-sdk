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

The `rest_managers` package contains clients for interacting with the various entities (
such as :py:class:`~sc_api_tools.data_models.project.Project`,
:py:class:`~sc_api_tools.data_models.media.Image` and
:py:class:`~sc_api_tools.data_models.model.Model`)
on the SC cluster.

All rest managers are initialized with a
:py:class:`~sc_api_tools.http_session.sc_session.SCSession` and a workspace id. The
:py:class:`~sc_api_tools.rest_managers.project_manager.project_manager.ProjectManager`
can be initialized with just that, while all other clients are initialized
*per project* and thus take an additional `project` argument.

For example, to initialize the
:py:class:`~sc_api_tools.rest_managers.media_managers.image_manager.ImageManager` for
a specific project and get a list of all images in the project, the following code
snippet can be used:

.. code-block:: python

   from sc_api_tools import SCRESTClient
   from sc_api_tools.rest_managers import ProjectManager, ImageManager

   client = SCRESTClient(
     host="https://0.0.0.0", username="dummy_user", password="dummy_password"
   )

   project_manager = ProjectManager(
       session=client.session, workspace_id=client.workspace_id
   )
   project = project_manager.get_project_by_name(project_name='dummy_project')

   image_manager = ImageManager(
       session=client.session, workspace_id=client.workspace_id, project=project
   )
   image_manager.get_all_images()

Module contents
---------------

.. autoclass:: sc_api_tools.rest_managers.project_manager.project_manager.ProjectManager
   :members:
   :undoc-members:

.. autoclass:: sc_api_tools.rest_managers.media_managers.image_manager.ImageManager
   :members:

.. autoclass:: sc_api_tools.rest_managers.media_managers.video_manager.VideoManager
   :members:

.. autoclass:: sc_api_tools.rest_managers.annotation_manager.annotation_manager.AnnotationManager
   :members:

.. autoclass:: sc_api_tools.rest_managers.configuration_manager.ConfigurationManager
   :members:

.. autoclass:: sc_api_tools.rest_managers.prediction_manager.PredictionManager
   :members:

.. autoclass:: sc_api_tools.rest_managers.model_manager.ModelManager
   :members:

.. autoclass:: sc_api_tools.rest_managers.training_manager.TrainingManager
   :members:

"""

from .annotation_manager import AnnotationManager
from .configuration_manager import ConfigurationManager
from .media_managers import ImageManager, VideoManager
from .model_manager import ModelManager
from .prediction_manager import PredictionManager
from .project_manager import ProjectManager
from .training_manager import TrainingManager

__all__ = [
    "AnnotationManager",
    "ConfigurationManager",
    "ProjectManager",
    "VideoManager",
    "ImageManager",
    "PredictionManager",
    "ModelManager",
    "TrainingManager",
]
