import copy
from typing import Optional, Sequence, Union, List, Dict, Any, Type, get_args

import attr

from sc_api_tools.data_models.configurable_parameter import (
    ConfigurableFloat,
    ConfigurableInteger,
    ConfigurableBoolean,
    SelectableFloat,
    SelectableString,
    ConfigurableParameter
)
from sc_api_tools.data_models.enums.configuration_enums import (
    ConfigurableParameterType,
    ParameterDataType,
    ParameterInputType
)
from sc_api_tools.data_models.utils import str_to_enum_converter, attr_value_serializer

PARAMETER_TYPES = Union[
    SelectableFloat,
    SelectableString,
    ConfigurableFloat,
    ConfigurableInteger,
    ConfigurableBoolean
]


def _parameter_dicts_to_list(
        parameter_dicts: List[Union[Dict[str, Any], ConfigurableParameter]]
) -> PARAMETER_TYPES:
    """
    Converts a list of dictionary representations of configurable parameters to a
    list of ConfigurableParameter objects

    :param parameter_dicts: List of dictionaries, each entry representing a single
        configurable parameter
    :return: List of corresponding ConfigurableParameter objects
    """
    parameters: List[PARAMETER_TYPES] = []
    for parameter in parameter_dicts:
        if isinstance(parameter, get_args(PARAMETER_TYPES)):
            parameters.append(parameter)
            continue
        data_type = parameter.get('data_type', None)
        template_type = parameter.get('template_type', None)
        if data_type is None or template_type is None:
            raise ValueError(
                f"Unable to reconstruct ParameterGroup object from input "
                f"dictionary: {input_dict}. No data or template type found for "
                f"parameter {parameter}"
            )
        data_type = ParameterDataType(data_type)
        template_type = ParameterInputType(template_type)
        parameter_type: Type[ConfigurableParameter]
        if (
                data_type == ParameterDataType.STRING
                and template_type == ParameterInputType.SELECTABLE
        ):
            parameter_type = SelectableString
        elif (
                data_type == ParameterDataType.INTEGER
                and template_type == ParameterInputType.INPUT
        ):
            parameter_type = ConfigurableInteger
        elif (
                data_type == ParameterDataType.BOOLEAN
                and template_type == ParameterInputType.INPUT
        ):
            parameter_type = ConfigurableBoolean
        elif (
                data_type == ParameterDataType.FLOAT
                and template_type == ParameterInputType.INPUT
        ):
            parameter_type = ConfigurableFloat
        elif (
                data_type == ParameterDataType.FLOAT
                and template_type == ParameterInputType.SELECTABLE
        ):
            parameter_type = SelectableFloat
        else:
            raise ValueError(
                f"Unable to reconstruct ParameterGroup object from input "
                f"dictionary: {input_dict}. Invalid data or template type found "
                f"for parameter {parameter}"
            )
        parameters.append(parameter_type(**parameter))
    return parameters


@attr.s(auto_attribs=True)
class ParameterGroup:
    """
    Class representing a group of configurable parameters in SC, as returned by the
    /configuration endpoints

    :var header: Human readable name for the parameter group
    :var type: Type of the parameter group
    :var name: name by which the parameter group is identified in the system
    :var parameters: List of configurable parameters
    :var groups: List of parameter groups
    """
    header: str
    type: str = attr.ib(converter=str_to_enum_converter(ConfigurableParameterType))
    description: str
    parameters: Optional[Sequence[PARAMETER_TYPES]] = None
    name: Optional[str] = None
    groups: Optional[List['ParameterGroup']] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the dictionary representation of the ParameterGroup

        :return:
        """
        return attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any]) -> 'ParameterGroup':
        """
        Creates a ParameterGroup object from an input dictionary

        :param input_dict: Dictionary to create the parameter group from
        :return: ParameterGroup instance corresponding to the data received in
            input_dict
        """
        input_copy = copy.deepcopy(input_dict)
        parameter_dicts: List[
            Union[Dict[str, Any], ConfigurableParameter]
        ] = input_copy.pop('parameters', [])
        group_dicts: List[
            Union[Dict[str, Any], ParameterGroup]
        ] = input_copy.pop('groups', [])

        parameters = _parameter_dicts_to_list(parameter_dicts=parameter_dicts)
        groups: List[ParameterGroup] = []
        for group_dict in group_dicts:
            if isinstance(group_dict, ParameterGroup):
                groups.append(group_dict)
                continue
            groups.append(ParameterGroup.from_dict(group_dict))
        return cls(**input_copy, parameters=parameters, groups=groups)

    def parameter_names(self, get_nested: bool = True) -> List[str]:
        """
        Returns a list of names of all parameters in the ParameterGroup

        :param get_nested: True to include parameter names in nested ParameterGroups
            belonging to this group as well. False to only consider this groups'
            parameters
        :return: List of parameter names
        """
        parameter_names = [parameter.name for parameter in self.parameters]
        if get_nested:
            for group in self.groups:
                parameter_names.extend(group.parameter_names(get_nested=True))
        return parameter_names

    def get_parameter_group_by_name(self, name: str) -> Optional['ParameterGroup']:
        """
        Returns the parameter group named `name`, if it belongs to this
        ParameterGroup or any of it's nested groups.

        If no group by `name` is found, this method returns None

        :param name: Name of the parameter group to look for
        :return: ParameterGroup named `name`, if any group by that name is found.
            None otherwise
        """
        all_groups = []
        for group in self.groups:
            all_groups.extend(group.groups)
        return next((group for group in all_groups if group.name == name), None)

    def get_parameter_by_name(
            self, name: str, group_name: Optional[str] = None
    ) -> Optional[ConfigurableParameter]:
        """
        Get the data for the configurable parameter named `name` from the ParameterGroup
        This method returns None if no parameter by that name was found

        :param name: Name of the parameter to get
        :param group_name: Optional name of the parameter group to which the parameter
            belongs. If None is specified (the default), this method looks in all
            parameter groups belonging to this group.
        :return: ConfigurableParameter object representing the parameter named `name`
        """
        if name not in self.parameter_names(get_nested=True):
            return None

        if group_name is None:
            parameters = [
                parameter for parameter in self.parameters if parameter.name == name
            ]
            for group in self.groups:
                parameters.extend(
                    [
                        parameter for parameter in group.parameters
                        if parameter.name == name
                    ]
                )
            if len(parameters) != 1:
                raise ValueError(
                    f"Found multiple parameters named {name}, please specify a "
                    f"group_name to identify the parameter unambiguously."
                )
            parameter = parameters[0]
        else:
            group = self.get_parameter_group_by_name(group_name)
            parameter = group.get_parameter_by_name(name)
        return parameter