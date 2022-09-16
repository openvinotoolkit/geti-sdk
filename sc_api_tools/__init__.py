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

These pages contain the documentation for the main SDK class,
:py:class:`~sc_api_tools.sc_rest_client.SCRESTClient`.

The :py:class:`~sc_api_tools.sc_rest_client.SCRESTClient` class implements convenience
methods for common operations that can be performed on the SC cluster, such as
creating a project from a pre-existing dataset, downloading or uploading a project,
uploading an image and getting a prediction for it and creating a deployment for a
project.

For example, to download a project, simply do:

.. code-block:: python

   from sc_api_tools import SCRESTClient

   client = SCRESTClient(
     host="https://0.0.0.0", username="dummy_user", password="dummy_password"
   )

   client.download_project(project_name="dummy_project")

For custom operations or more fine-grained control over the behavior, the
:py:mod:`~sc_api_tools.rest_clients` subpackage should be used.

Module contents
---------------

.. autoclass:: sc_api_tools.sc_rest_client::SCRESTClient
   :no-members:

   .. rubric:: Project download and upload

   .. automethod:: download_project

   .. automethod:: upload_project

   .. automethod:: download_all_projects

   .. automethod:: upload_all_projects

   .. rubric:: Project creation from dataset

   .. automethod:: create_single_task_project_from_dataset

   .. automethod:: create_task_chain_project_from_dataset

   .. rubric:: Project deployment

   .. automethod:: deploy_project

   .. rubric:: Media upload and prediction

   .. automethod:: upload_and_predict_image

   .. automethod:: upload_and_predict_video

   .. automethod:: upload_and_predict_media_folder

"""

from .sc_rest_client import SCRESTClient

name = "sc-api-tools"

__version__ = "0.1.5"

__all__ = ["SCRESTClient"]
