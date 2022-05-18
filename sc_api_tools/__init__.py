"""
SonomaCreek SDK
===============

Welcome to the documentation for the SonomaCreek REST SDK! These pages contain the
documentation for the main SDK class,
:py:class:`~sc_api_tools.sc_rest_client.SCRESTClient`, as well as for the subpackages
in the SDK.

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
:py:mod:`~sc_api_tools.rest_managers` subpackage should be used.

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

name = 'sc-api-tools'

__version__ = '0.0.1'

__all__ = ["SCRESTClient"]
