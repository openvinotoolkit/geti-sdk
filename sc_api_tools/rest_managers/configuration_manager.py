from typing import Union, Optional

from sc_api_tools.data_models import (
    Project,
    TaskConfiguration,
    GlobalConfiguration
)
from sc_api_tools.data_models.configurable_parameter_group import PARAMETER_TYPES
from sc_api_tools.http_session import SCSession
from sc_api_tools.rest_converters import ConfigurationRESTConverter


class ConfigurationManager:
    """
    Class to manage configuration for a certain project
    """

    def __init__(self, workspace_id: str, project: Project, session: SCSession):
        self.session = session
        project_id = project.id
        self.task_ids = [task.id for task in project.get_trainable_tasks()]
        self.base_url = f"workspaces/{workspace_id}/projects/{project_id}/" \
                        f"configuration/"

    def get_task_configuration(self, task_id: str) -> TaskConfiguration:
        """
        Gets the configuration for the task with id `task_id`

        :param task_id: ID of the task to get configurations for
        :return: TaskConfiguration holding all component parameters and hyper parameters
            for the task
        """
        config_data = self.session.get_rest_response(
            url=f"{self.base_url}task_chain/{task_id}",
            method="GET"
        )
        return ConfigurationRESTConverter.task_configuration_from_dict(config_data)

    def get_global_configuration(self) -> GlobalConfiguration:
        """
        Gets the project-wide configurable parameters

        :return: GlobalConfiguration instance holding the configurable parameters for
            all project-wide components
        """
        config_data = self.session.get_rest_response(
            url=f"{self.base_url}global",
            method="GET"
        )
        return ConfigurationRESTConverter.global_configuration_from_rest(config_data)

    def set_task_configuration(self, task_id: str, config: dict):
        """
        Update the configuration for a task

        :param task_id: ID of the task to set the configuration for
        :param config: Dictionary containing the updated configuration values
        :return: Response of the configuration POST endpoint.
        """
        response = self.session.get_rest_response(
            url=f"{self.base_url}task_chain/{task_id}",
            method="POST",
            data=config
        )
        return response

    def set_project_auto_train(self, auto_train: bool = False) -> None:
        """
        Sets the `auto_train` parameter for all tasks in the project

        :param auto_train: True to enable auto_training, False to disable
        """
        for task_id in self.task_ids:
            config = self.get_task_configuration(task_id=task_id)
            config_data = config.set_parameter_value('auto_training', value=auto_train)
            self.set_task_configuration(task_id=task_id, config=config_data)

    def set_project_num_iterations(self, value: int = 50):
        """
        Sets the number of iterations to train for each task in the project

        :param value: Number of iterations to set
        """
        iteration_names = ["num_iters", "max_num_epochs"]
        for task_id in self.task_ids:
            config = self.get_task_configuration(task_id=task_id)
            parameter: Optional[PARAMETER_TYPES] = None
            for parameter_name in iteration_names:
                parameter = config.get_parameter_by_name(parameter_name)
                if parameter is not None:
                    self.set_task_configuration(
                        task_id=task_id,
                        config=config.set_parameter_value(parameter_name, value=value)
                    )
                    break
            if parameter is None:
                raise ValueError(
                    f"No iteration parameters were found for task {config.task_title}. "
                    f"Unable to set number of iterations"
                )

    def set_project_parameter(
            self,
            parameter_name: str,
            value: Union[bool, str, float, int],
            parameter_group_name: Optional[str] = None
    ):
        """
        Sets the value for a parameter with `parameter_name` that lives in the
        group `parameter_group_name`. The parameter is set for all tasks in the project

        The `parameter_group_name` can be left as None, in that case this method will
        attempt to determine the appropriate parameter group automatically.

        :param parameter_name: Name of the parameter
        :param parameter_group_name: Optional name of the parameter group name to
            which the parameter belongs. If left as None (the default), this method will
            attempt to determine the correct parameter group automatically, if needed.
        :param value: Value to set for the parameter
        """
        for index, task_id in enumerate(self.task_ids):
            config = self.get_task_configuration(task_id)
            config_data = config.set_parameter_value(
                parameter_name=parameter_name,
                value=value,
                group_name=parameter_group_name
            )
            self.set_task_configuration(task_id=task_id, config=config_data)
