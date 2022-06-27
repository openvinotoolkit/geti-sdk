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

.. warning:: This package is experimental, meaning that it's contents are subject to
    change and it may not provide full support for all possible SC models. Please test
    carefully before relying on this in a production environment.

The `deployment` package allows creating a deployment of an SC project. A project
deployment can run inference on an image or video frame locally, i.e. without any
connect to the SC cluster.

Deployments can be created for both single task and task chain projects alike, the API
is the same in both cases.

Creating a deployment for a project is done through the
:py:class:`~sc_api_tools.sc_rest_client.SCRESTClient` class, which provides a
convenience method :py:meth:`~sc_api_tools.sc_rest_client.SCRESTClient.deploy_project`.

The following code snippet shows:

 #. How to create a deployment for a project
 #. How to use it to run local inference for an image
 #. How to save the deployment to disk

.. code-block:: python

   import cv2

   from sc_api_tools import SCRESTClient

   client = SCRESTClient(
     host="https://0.0.0.0", username="dummy_user", password="dummy_password"
   )

   # Download the model data and create a `Deployment`
   deployment = client.deploy_project(project_name="dummy_project")

   # Load the inference models for all tasks in the project, for CPU inference
   deployment.load_inference_models(device='CPU')

   # Run inference
   dummy_image = cv2.imread('dummy_image.png')
   dummy_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
   prediction = deployment.infer(image=dummy_image)

   # Save the deployment to disk
   deployment.save(path_to_folder="deployment_dummy_project")

A saved Deployment can be loaded from its containing folder using the
:py:meth:`~sc_api_tools.deployment.deployment.Deployment.from_folder` method, like so:

.. code-block:: python

   from sc_api_tools.deployment import Deployment

   local_deployment = Deployment.from_folder("deployment_dummy_project")

Module contents
---------------

.. automodule:: sc_api_tools.deployment.deployed_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sc_api_tools.deployment.deployment
   :members:
   :undoc-members:
   :show-inheritance:
"""

from .deployment import Deployment
from .deployed_model import DeployedModel

__all__ = ["Deployment", "DeployedModel"]
