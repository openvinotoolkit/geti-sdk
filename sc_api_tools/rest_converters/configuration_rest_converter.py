import copy
from typing import Dict, Any, List, Union

from sc_api_tools.data_models.configuration import (
    ConfigurableParameters,
    TaskConfiguration, GlobalConfiguration
)
from sc_api_tools.data_models.configuration_identifiers import (
    EntityIdentifier,
    HyperParameterGroupIdentifier,
    ComponentEntityIdentifier
)
from sc_api_tools.data_models.enums.configuration_enums import (
    ConfigurationEntityType
)


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
        component_objects: List[ConfigurableParameters] = []
        for component in components:
            if not isinstance(component, ConfigurableParameters):
                component_objects.append(
                    ConfigurationRESTConverter.from_dict(component)
                )
            else:
                component_objects.append(component)
        input_copy.update({'components': component_objects})
        return TaskConfiguration(**input_copy)

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
        component_objects: List[ConfigurableParameters] = []
        if isinstance(input_copy, list):
            input_list = input_copy
        else:
            input_list = input_copy.pop("components")
        for component in input_list:
            if not isinstance(component, ConfigurableParameters):
                component_objects.append(
                    ConfigurationRESTConverter.from_dict(component)
                )
            else:
                component_objects.append(component)
        if isinstance(input_copy, list):
            return GlobalConfiguration(components=component_objects)
        else:
            input_copy.update({'components': component_objects})
            return GlobalConfiguration(**input_copy)
