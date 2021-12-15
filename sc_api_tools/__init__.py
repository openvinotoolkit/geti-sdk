from .annotation_readers import (
    VitensAnnotationReader,
    DatumAnnotationReader,
    SCAnnotationReader
)
from .http_session import (
    SCSession,
    ClusterConfig
)
from .rest_managers import (
    ProjectManager,
    ConfigurationManager,
    MediaManager,
    AnnotationManager
)
from .utils import get_default_workspace_id
from .sc_rest_client import SCRESTClient

name = 'sc-api-tools'
__version__ = '0.0.1'

__all__ = [
    "SCSession",
    "ClusterConfig",
    "ProjectManager",
    "ConfigurationManager",
    "MediaManager",
    "AnnotationManager",
    "DatumAnnotationReader",
    "VitensAnnotationReader",
    "SCAnnotationReader",
    "get_default_workspace_id",
    "SCRESTClient"
]
