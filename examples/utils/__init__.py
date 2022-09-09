# noqa: D104

import os

from .project_utilities import ensure_example_project

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "notebooks", "data"
)

__all__ = ["ensure_example_project", "DATA_PATH"]
