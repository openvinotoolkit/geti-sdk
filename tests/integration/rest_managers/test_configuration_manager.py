from typing import List

import pytest

from tests.helpers.project_service import ProjectService
from tests.helpers.constants import PROJECT_PREFIX


class TestConfigurationManager:
    @pytest.mark.vcr()
    def test_get_and_set_configuration(
            self,
            fxt_project_service: ProjectService,
            fxt_default_labels: List[str]
    ):
        """
        Verifies that getting and setting the configuration for a single task project
        works as expected

        Steps:
        1. Create detection project
        2. Get task configuration
        3. Assert that 'batch_size' is part of the task configuration
        4. Update the configuration so that the new batch size is half the old batch
            size
        5. POST the new configuration to the server
        6. GET the task configuration again, and assert that the batch size has changed
        """
        project = fxt_project_service.create_project(
            project_name=f"{PROJECT_PREFIX}_configuration_manager",
            project_type="detection",
            labels=[fxt_default_labels]
        )
        task = project.get_trainable_tasks()[0]

        configuration_manager = fxt_project_service.configuration_manager
        task_configuration = configuration_manager.get_task_configuration(task.id)
        assert "batch_size" in task_configuration.get_all_parameter_names()
        old_batch_size = task_configuration.batch_size.value
        new_batch_size = int(old_batch_size / 2)
        task_configuration.set_parameter_value("batch_size", new_batch_size)
        configuration_manager.set_configuration(task_configuration)

        new_task_configuration = configuration_manager.get_task_configuration(task.id)
        assert new_task_configuration.batch_size.value == new_batch_size
