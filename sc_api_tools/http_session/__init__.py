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

The `http_session` package acts as a wrapper around `requests.Session` and handles all
http requests to the SC cluster.

Normally it is not necessary to interact with the classes in this package directly, as
they are instantiated through the main :py:class:`sc_api_tools.SCRESTClient` class.

However, an SCSession can be established directly using the following code snippet:

.. code-block:: python

   from sc_api_tools.http_session import ClusterConfig, SCSession

   config = ClusterConfig(
     host="https://0.0.0.0", username="dummy_user", password="dummy_password"
   )
   session = SCSession(cluster_config=config)

Initializing the session will perform authentication on the SC cluster.

Module contents
---------------

.. autoclass:: sc_api_tools.http_session.sc_session.SCSession
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: sc_api_tools.http_session.cluster_config.ClusterConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: sc_api_tools.http_session.exception.SCRequestException
   :members:
   :undoc-members:
   :show-inheritance:

"""

from .sc_session import SCSession
from .cluster_config import ClusterConfig
from .exception import SCRequestException

__all__ = ["SCSession", "ClusterConfig", "SCRequestException"]
