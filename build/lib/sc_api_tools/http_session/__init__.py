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

"""

from .sc_session import SCSession
from .cluster_config import ClusterConfig

__all__ = ["SCSession", "ClusterConfig"]
