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

import pytest

from geti_sdk.data_models import Algorithm, TaskType
from geti_sdk.utils import get_supported_algorithms


class TestUtils:
    @pytest.mark.vcr()
    def test_get_supported_algorithms(self, fxt_geti_session):
        """
        Verifies that getting the list of supported algorithms from the server works
        as expected

        Test steps:
        1. Retrieve a list of supported algorithms from the server
        2. Assert that the returned list is not emtpy
        3. Assert that each entry in the list is a properly deserialized Algorithm
            instance
        4. Filter the AlgorithmList to select only the classification algorithms from
            it
        5. Assert that the list of classification algorithms is not empty and that
            each algorithm in it has the proper task type
        """
        algorithms = get_supported_algorithms(fxt_geti_session)

        assert len(algorithms) > 0
        for algorithm in algorithms:
            assert isinstance(algorithm, Algorithm)

        classification_algos = algorithms.get_by_task_type(
            task_type=TaskType.CLASSIFICATION
        )
        assert len(classification_algos) > 0
        for algorithm in classification_algos:
            assert algorithm.task_type == TaskType.CLASSIFICATION
