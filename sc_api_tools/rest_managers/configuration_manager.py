import copy

from sc_api_tools.data_models import Project
from sc_api_tools.http_session import SCSession


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

    def get_task_configuration(self, task_id: str):
        config_data = self.session.get_rest_response(
            url=f"{self.base_url}task_chain/{task_id}",
            method="GET"
        )
        return config_data["components"]

    def get_component_configurations(self, task_id: str):
        task_config = self.get_task_configuration(task_id)
        component_configs = [
            config for config in task_config
            if config["entity_identifier"]["type"] == "COMPONENT_PARAMETERS"
        ]
        return component_configs

    def set_task_configuration(self, task_id: str, config: dict):
        response = self.session.get_rest_response(
            url=f"{self.base_url}task_chain/{task_id}",
            method="POST",
            data=config
        )
        return response

    def set_project_auto_train(self, auto_train: bool = False):
        for task_id in self.task_ids:
            config = self.get_component_configurations(task_id)
            general_parameters = next(
                (parameters for parameters in config
                 if parameters["entity_identifier"]["component"] == "TASK_NODE")
            )
            entity_identifier = copy.deepcopy(general_parameters["entity_identifier"])
            config_data = {
                "components": [
                    {
                        "entity_identifier": entity_identifier,
                        "parameters": [
                            {
                                "name": "auto_training",
                                "value": auto_train
                            }
                        ]
                    }
                ]
            }
            self.set_task_configuration(task_id=task_id, config=config_data)

    def set_project_num_iterations(self, value: int = 50):
        """
        Sets the number of iterations to train for the project
        :param value:
        :return:
        """
        learning_parameter_group_names = ["learning_parameters", "dataset"]
        iteration_names = ["num_iters", "max_num_epochs"]
        for task_id in self.task_ids:
            config = self.get_task_configuration(task_id)
            learning_parameters = next(
                (item for item in config
                 if item["name"] in learning_parameter_group_names), None
            )
            if learning_parameters is None:
                raise ValueError(
                    "Unable to determine learning parameter group from task "
                    "configuration. Aborting"
                )
            learning_parameter_group_entity_identifier = learning_parameters[
                "entity_identifier"
            ]

            config_data = {
                "components": [
                    {
                    "entity_identifier": learning_parameter_group_entity_identifier,
                    "parameters": []
                    }
                ]
            }
            for name in iteration_names:
                if name in learning_parameters["parameters"]:
                    config_data["components"][0]["parameters"].append(
                        {
                            "name": name,
                            "value": value
                        }
                    )
            self.set_task_configuration(task_id=task_id, config=config_data)
