import copy
from typing import Dict, Any, List, Union

import attr

from sc_api_tools.data_models.configurable_parameter_group import ParameterGroup
from sc_api_tools.data_models.configuration import (
    ConfigurableParameters,
    TaskConfiguration,
    GlobalConfiguration,
    FullConfiguration
)
from sc_api_tools.data_models.configuration_identifiers import (
    EntityIdentifier,
    HyperParameterGroupIdentifier,
    ComponentEntityIdentifier
)
from sc_api_tools.data_models.enums.configuration_enums import (
    ConfigurationEntityType
)
from sc_api_tools.utils import remove_null_fields


class ConfigurationRESTConverter:
    """
    Class that handles conversion of SC REST output for configurable parameter
    entities to objects and vice versa.
    """
    @staticmethod
    def entity_identifier_from_dict(input_dict: Dict[str, Any]) -> EntityIdentifier:
        """
        Creates an EntityIdentifier object from an input dictionary

        :param input_dict: Dictionary representing an EntityIdentifier in SC
        :return: EntityIdentifier object corresponding to the data in `input_dict`
        """
        identifier_type = input_dict.get("type", None)
        if isinstance(identifier_type, str):
            identifier_type = ConfigurationEntityType(identifier_type)
        if identifier_type == ConfigurationEntityType.HYPER_PARAMETER_GROUP:
            identifier_class = HyperParameterGroupIdentifier
        elif identifier_type == ConfigurationEntityType.COMPONENT_PARAMETERS:
            identifier_class = ComponentEntityIdentifier
        else:
            raise ValueError(
                f"Invalid entity identifier type found: Entity identifier of type "
                f"{identifier_type} is not supported."
            )
        return identifier_class(**input_dict)

    @staticmethod
    def from_dict(input_dict: Dict[str, Any]) -> ConfigurableParameters:
        """
        Creates a ConfigurableParameters object holding the configurable parameters
        for an entity in SC, from a dictionary returned by the SC /configuration REST
        endpoints

        :param input_dict: Dictionary containing the configurable parameters
        :return: ConfigurableParameters instance holding the parameter data
        """
        input_copy = copy.deepcopy(input_dict)
        entity_identifier = input_copy.pop("entity_identifier")

        if not isinstance(entity_identifier, EntityIdentifier):
            entity_identifier = ConfigurationRESTConverter.entity_identifier_from_dict(
                input_dict=entity_identifier
            )

        input_copy['entity_identifier'] = entity_identifier
        return ConfigurableParameters.from_dict(input_dict=input_copy)

    @staticmethod
    def _rest_components_to_objects(
            input_list: List[Dict[str, Any]]
    ) -> List[ConfigurableParameters]:
        """
        Creates a list of configurable parameters from a list of dictionaries received
        by the SC /configuration endpoints

        :param input_list: List of dictionaries to convert
        :return: List of ConfigurableParameters instances
        """
        component_objects: List[ConfigurableParameters] = []
        for component in input_list:
            if not isinstance(component, ConfigurableParameters):
                component_objects.append(
                    ConfigurationRESTConverter.from_dict(component)
                )
            else:
                component_objects.append(component)
        return component_objects

    @staticmethod
    def _remove_non_minimal_fields(configurable_parameters: ParameterGroup):
        """
        For all parameters in the parameter group, sets the fields that are not part
        of the minimal representation of the parameter to 'None'

        NOTE: This method modifies the input in place

        :param configurable_parameters: ConfigurableParameters instance to convert to
            minimal representation
        :return:
        """
        # Set non minimal fields of ConfigurableParameters object to None
        for field in attr.fields(type(configurable_parameters)):
            if field.name in configurable_parameters._non_minimal_fields:
                setattr(configurable_parameters, field.name, None)
        # Set non minimal fields of individual parameters to None
        for parameter in configurable_parameters.parameters:
            for field in attr.fields(type(parameter)):
                if field.name in parameter._non_minimal_fields:
                    setattr(parameter, field.name, None)
        # Set groups to None if no groups, else remove redundant fields from all groups
        if not configurable_parameters.groups:
            configurable_parameters.groups = None
        else:
            for group in configurable_parameters.groups:
                ConfigurationRESTConverter._remove_non_minimal_fields(group)

    @staticmethod
    def task_configuration_from_dict(input_dict: Dict[str, Any]) -> TaskConfiguration:
        """
        Creates a TaskConfiguration object holding all configurable parameters for a
        task in SC, from a dictionary returned by the /configuration REST endpoints

        :param input_dict: Dictionary containing the configurable parameters for the
            task
        :return: TaskConfiguration instance holding the parameter data
        """
        input_copy = copy.deepcopy(input_dict)
        components = input_copy.pop("components")
        component_objects = ConfigurationRESTConverter._rest_components_to_objects(
            components
        )
        input_copy.update({'components': component_objects})
        return TaskConfiguration(**input_copy)

    @staticmethod
    def configuration_to_minimal_dict(
            configuration: Union[
                TaskConfiguration, GlobalConfiguration, FullConfiguration
            ],
            deidentify: bool = True
    ) -> Dict[str, Any]:
        """
        Converts a TaskConfiguration, GlobalConfiguration or FullConfiguration into a
        dictionary, removing fields that are None or are only relevant to the SC UI

        :param configuration: TaskConfiguration or GlobalConfiguration to convert
        :param deidentify: True to remove all unique database identifiers, False to
            preserve identifiers. Defaults to True
        :return: Dictionary representation of the configuration
        """
        input_copy = copy.deepcopy(configuration)
        if deidentify:
            input_copy.deidentify()
        if not isinstance(configuration, FullConfiguration):
            for config in input_copy.components:
                ConfigurationRESTConverter._remove_non_minimal_fields(config)
        else:
            for config in input_copy.global_.components:
                ConfigurationRESTConverter._remove_non_minimal_fields(config)
            for task_config in input_copy.task_chain:
                for config in task_config.components:
                    ConfigurationRESTConverter._remove_non_minimal_fields(config)
        result = input_copy.to_dict()
        remove_null_fields(result)
        return result

    @staticmethod
    def global_configuration_from_rest(
            input_: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> GlobalConfiguration:
        """
        Creates a GlobalConfiguration object holding the configurable parameters
        for all project-wide components in the SC project, from input from the
        /configuration/global REST endpoint

        :param input_: REST response holding the serialized configurable parameters
        :return:
        """
        input_copy = copy.deepcopy(input_)
        if isinstance(input_copy, list):
            input_list = input_copy
        else:
            input_list = input_copy.pop("components")
        component_objects = ConfigurationRESTConverter._rest_components_to_objects(
            input_list
        )
        if isinstance(input_copy, list):
            return GlobalConfiguration(components=component_objects)
        else:
            input_copy.update({'components': component_objects})
            return GlobalConfiguration(**input_copy)

    @staticmethod
    def full_configuration_from_rest(input_dict: Dict[str, Any]) -> FullConfiguration:
        """
        Converts a dictionary holding the full configuration for an SC project, as
        returned by the /configuration endpoint, to an object representation

        :param input_dict: Dictionary representing the full project configuration
        :return: FullConfiguration instance holding the global and task chain
            configuration
        """
        global_dict = input_dict.pop("global")
        task_chain_list = input_dict.pop("task_chain")
        global_config = ConfigurationRESTConverter.global_configuration_from_rest(
            global_dict
        )
        task_chain_config = [
            ConfigurationRESTConverter.task_configuration_from_dict(task_config)
            for task_config in task_chain_list
        ]
        return FullConfiguration(global_=global_config, task_chain=task_chain_config)

    @staticmethod
    def configurable_parameter_list_to_rest(
            configurable_parameter_list: List[ConfigurableParameters]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Converts a list of model hyper parameters to a dictionary that can be sent to
        the /configuration POST endpoints

        :param configurable_parameter_list: List of ConfigurableParameter instances
        :return: Dictionary containing:
            - 'components': list of dictionaries representing configurable parameters,
                            that are conforming to the /configuration REST endpoints
        """
        rest_parameters: List[Dict[str, Any]] = []
        for parameter_set in configurable_parameter_list:
            parameter_copy = copy.deepcopy(parameter_set)
            ConfigurationRESTConverter._remove_non_minimal_fields(
                parameter_copy
            )
            parameter_dict = parameter_copy.to_dict()
            remove_null_fields(parameter_dict)
            rest_parameters.append(parameter_dict)
        return {'components': rest_parameters}
