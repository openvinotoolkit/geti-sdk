from typing import ClassVar, Dict, Any, Optional

import attr

from sc_api_tools.data_models.enums import ConfigurationEntityType
from sc_api_tools.data_models.utils import str_to_enum_converter, attr_value_serializer


@attr.s(auto_attribs=True)
class EntityIdentifier:
    """
    Class representing identification information for a configurable entity in SC, as
    returned by the /configuration endpoint

    :var workspace_id: ID of the workspace to which the entity belongs
    :var type: Type of the configuration
    """
    _identifier_fields: ClassVar[str] = ["workspace_id"]

    workspace_id: str
    type: str = attr.ib(converter=str_to_enum_converter(ConfigurationEntityType))

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the dictionary representation of the EntityIdentifier

        :return: Dictionary representing the EntityIdentifier
        """
        return attr.asdict(self, value_serializer=attr_value_serializer, recurse=True)


@attr.s(auto_attribs=True)
class HyperParameterGroupIdentifier(EntityIdentifier):
    """
    Class representing the identification information for a HyperParameterGroup in SC,
    as returned by the /configuration endpoint

    :var model_storage_id: ID of the model storage to which the hyper parameter group
        belongs
    :var group_name: Name of the hyper parameter group
    """
    _identifier_fields: ClassVar[str] = ["model_storage_id", "workspace_id"]

    model_storage_id: str
    group_name: str


@attr.s(auto_attribs=True)
class ComponentEntityIdentifier(EntityIdentifier):
    """
    Class representing the identification information for a configurable Component in
    SC, as returned by the /configuration endpoint.

    :var component: Name of the component
    :var project_id: ID of the project to which the component belongs
    :var task_id: Optional ID of the task to which the component belongs
    """
    _identifier_fields: ClassVar[str] = ["project_id", "task_id", "workspace_id"]

    component: str
    project_id: str
    task_id: Optional[str] = None