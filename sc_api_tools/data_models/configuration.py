from typing import Optional, Union, ClassVar

import attr
from sc_api_tools.data_models.configurable_parameter_group import ParameterGroup
from sc_api_tools.data_models.configuration_identifiers import \
    HyperParameterGroupIdentifier, ComponentEntityIdentifier

from sc_api_tools.data_models.utils import deidentify


@attr.s(auto_attribs=True)
class ConfigurableParameters(ParameterGroup):
    """
    Class representing configurable parameters in SC, as returned by the
    /configuration endpoint

    :var entity_identifier: Identification information for the entity to which the
        configurable parameters apply
    :var id: Unique database ID of the configurable parameters
    """
    _identifier_fields: ClassVar[str] = ["id"]

    entity_identifier: Union[
        HyperParameterGroupIdentifier, ComponentEntityIdentifier
    ] = attr.ib(kw_only=True)
    id: Optional[str] = attr.ib(default=None, kw_only=True)

    def deidentify(self):
        """
        Removes all unique database ID's from the configurable parameters

        """
        deidentify(self)
        deidentify(self.entity_identifier)
