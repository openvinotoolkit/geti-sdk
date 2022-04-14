import os
from typing import Callable

import pytest

from sc_api_tools.annotation_readers import DatumAnnotationReader


@pytest.fixture(scope="session")
def fxt_blocks_dataset(fxt_base_test_path) -> str:
    """
    This fixture returns the path to the 'blocks' dataset
    """
    yield os.path.join(fxt_base_test_path, "data", "blocks")


@pytest.fixture(scope="session")
def fxt_image_folder(fxt_blocks_dataset) -> str:
    """
    This fixture returns the path to a sample image
    """
    yield os.path.join(fxt_blocks_dataset, "images", "NONE")


@pytest.fixture(scope="session")
def fxt_image_path(fxt_image_folder) -> str:
    """
    This fixture returns the path to a sample image
    """
    yield os.path.join(fxt_image_folder, "WIN_20220406_21_24_24_Pro.jpg")


@pytest.fixture(scope="function")
def fxt_annotation_reader(fxt_blocks_dataset) -> DatumAnnotationReader:
    """
    This fixture returns a Datumaro Annotation Reader which can read annotations for
    the 'blocks' dataset
    """
    yield DatumAnnotationReader(
        base_data_folder=fxt_blocks_dataset,
        annotation_format='coco'
    )


@pytest.fixture(scope="function")
def fxt_annotation_reader_factory(
    fxt_blocks_dataset
) -> Callable[[None], DatumAnnotationReader]:
    """
    This fixutre returns Datumaro Annotation Readers which can read annotations for
    the 'blocks' dataset. The fixture can be called multiple times to yield different
    instances of the annotation reader
    """
    def _create_annotation_reader() -> DatumAnnotationReader:
        return DatumAnnotationReader(
            base_data_folder=fxt_blocks_dataset,
            annotation_format='coco'
        )
    yield _create_annotation_reader
