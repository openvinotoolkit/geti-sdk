import pytest

from sc_api_tools.data_models import Algorithm, TaskType
from sc_api_tools.utils import get_supported_algorithms


class TestUtils:
    @pytest.mark.vcr()
    def test_get_supported_algorithms(self, fxt_sc_session):
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
        algorithms = get_supported_algorithms(fxt_sc_session)

        assert len(algorithms) > 0
        for algorithm in algorithms:
            assert isinstance(algorithm, Algorithm)

        classification_algos = algorithms.get_by_task_type(
            task_type=TaskType.CLASSIFICATION
        )
        assert len(classification_algos) > 0
        for algorithm in classification_algos:
            assert algorithm.task_type == TaskType.CLASSIFICATION
