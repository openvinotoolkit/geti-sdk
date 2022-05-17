"""
Introduction
------------

The `annotation_readers` package contains the
:py:class:`~sc_api_tools.annotation_readers.base_annotation_reader.AnnotationReader` base
class, which provides an interface for implementing custom annotation readers.

Annotation readers server to load annotation files in custom formats and convert them
to SC format, such that they can be uploaded to an SC project.

Module contents
---------------

.. automodule:: sc_api_tools.annotation_readers.base_annotation_reader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sc_api_tools.annotation_readers.sc_annotation_reader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sc_api_tools.annotation_readers.datumaro_annotation_reader.datumaro_annotation_reader
   :members:
   :undoc-members:
   :show-inheritance:
"""

from .base_annotation_reader import AnnotationReader
from .vitens_annotation_reader import VitensAnnotationReader
from .datumaro_annotation_reader import DatumAnnotationReader
from .sc_annotation_reader import SCAnnotationReader
from .nous_annotation_reader import NOUSAnnotationReader

__all__ = [
    "AnnotationReader",
    "VitensAnnotationReader",
    "DatumAnnotationReader",
    "SCAnnotationReader",
    "NOUSAnnotationReader"
]
