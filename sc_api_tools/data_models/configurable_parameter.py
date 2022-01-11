from typing import Union, Optional, Dict, Any, List

import attr

from sc_api_tools.data_models.enums.configuration_enums import (
    ParameterDataType,
    ParameterInputType
)
from sc_api_tools.data_models.utils import str_to_enum_converter, attr_value_serializer


@attr.s(auto_attribs=True)
class ConfigurableParameter:
    """
    Class representing a single configurable parameter in SC

    :var data_type: Data type for the parameter. Can be integer, float, string or
        boolean
    :var default_value: Default value for the parameter
    :var description: Human readable description of the parameter
    :var editable: Boolean indicating whether this parameter is editable (True) or not
        (False)
    :var header: Human readable name for the parameter
    :var name: system name for the parameter
    :var template_type: Indicates whether the parameter takes free input (`input`)
        or the value has to be selected from a list of options (`selectable`)
    :var value: The current value for the parameter
    :var ui_rules: Dictionary representing rules for logic processing in the UI,
        based on parameter values
    :var warning: Optional warning message pointing out possible risks of changing the
        parameter
    """
    name: str
    value: Union[str, bool, float, int]
    data_type: Optional[str] = attr.ib(
        default=None, converter=str_to_enum_converter(ParameterDataType)
    )
    default_value: Optional[Union[str, bool, float, int]] = None
    description: Optional[str] = None
    editable: Optional[bool] = None
    header: Optional[str] = None
    template_type: Optional[str] = attr.ib(
        default=None, converter=str_to_enum_converter(ParameterInputType)
    )
    ui_rules: Optional[Dict[str, Any]] = None
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the dictionary representation of the ConfigurableParameter object

        :return:
        """
        return attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)


@attr.s(auto_attribs=True)
class ConfigurableBoolean(ConfigurableParameter):
    """
    Class representing a configurable boolean in SC
    """
    default_value: Optional[bool] = attr.ib(default=None,kw_only=True)
    value: bool = attr.ib(kw_only=True)


@attr.s(auto_attribs=True)
class ConfigurableInteger(ConfigurableParameter):
    """
    Class representing a configurable integer in SC

    :var min_value: Minimum value allowed to be set for the configurable integer
    :var max_value: Maximum value allowed to be set for the configurable integer
    """
    default_value: Optional[int] = attr.ib(default=None, kw_only=True)
    value: int = attr.ib(kw_only=True)
    min_value: Optional[int] = attr.ib(default=None, kw_only=True)
    max_value: Optional[int] = attr.ib(default=None, kw_only=True)


@attr.s(auto_attribs=True)
class ConfigurableFloat(ConfigurableParameter):
    """
    Class representing a configurable float in SC

    :var min_value: Minimum value allowed to be set for the configurable float
    :var max_value: Maximum value allowed to be set for the configurable float
    """
    default_value: float = attr.ib(kw_only=True)
    value: float = attr.ib(kw_only=True)
    min_value: float = attr.ib(kw_only=True)
    max_value: float = attr.ib(kw_only=True)


@attr.s(auto_attribs=True)
class SelectableFloat(ConfigurableParameter):
    """
    Class representing a float selectable in SC

    :var options: List of options that the selectable float is allowed to take
    """
    default_value: float = attr.ib(kw_only=True)
    value: float = attr.ib(kw_only=True)
    options: List[float] = attr.ib(kw_only=True)


@attr.s(auto_attribs=True)
class SelectableString(ConfigurableParameter):
    """
    Class representing a string selectable in SC

    :var options: List of options that the selectable string is allowed to take
    """
    default_value: str = attr.ib(kw_only=True)
    enum_name: str = attr.ib(kw_only=True)
    value: str = attr.ib(kw_only=True)
    options: List[str] = attr.ib(kw_only=True)
