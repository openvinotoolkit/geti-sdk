from typing import List, Dict

import pytest


@pytest.fixture()
def fxt_hierarchical_classification_labels() -> List[Dict[str, str]]:
    yield [
            {"name": "animal"},
            {"name": "dog", "parent_id": "animal"},
            {"name": "cat", "parent_id": "animal"},
            {"name": "vehicle"},
            {"name": "car", "parent_id": "vehicle", "group": "vehicle type"},
            {"name": "taxi", "parent_id": "vehicle", "group": "vehicle type"},
            {"name": "truck", "parent_id": "vehicle", "group": "vehicle type"},
            {"name": "red", "parent_id": "vehicle", "group": "vehicle color"},
            {"name": "blue", "parent_id": "vehicle", "group": "vehicle color"},
            {"name": "black", "parent_id": "vehicle", "group": "vehicle color"},
            {"name": "grey", "parent_id": "vehicle", "group": "vehicle color"}
    ]


@pytest.fixture()
def fxt_default_labels() -> List[str]:
    yield ["cube", "cylinder"]
