from pprint import pformat
from typing import Optional, List, Dict, Any

import attr

from sc_api_tools.data_models.enums import ModelStatus, OptimizationType
from sc_api_tools.data_models.utils import str_to_datetime, str_to_enum_converter, \
    attr_value_serializer


@attr.s(auto_attribs=True)
class OptimizationCapabilities:
    """
    Class representing model optimization capabilities in SC
    """
    is_filter_pruning_enabled: bool
    is_nncf_supported: bool


@attr.s(auto_attribs=True)
class BaseModel:
    """
    Class representing the basic information about a Model or OptimizedModel in SC
    """
    name: str
    fps_throughput: str
    latency: str
    precision: List[str]
    creation_date: str = attr.ib(converter=str_to_datetime)
    target_device: Optional[str] = None
    target_device_type: Optional[str] = None
    previous_revision_id: Optional[str] = None
    previous_trained_revision_id: Optional[str] = None
    score: Optional[float] = attr.ib(default=None)
    id: Optional[str] = attr.ib(default=None)

    def __attrs_post_init__(self):
        self._model_group_id: Optional[str] = None

    @property
    def model_group_id(self) -> Optional[str]:
        """
        Returns the unique database ID of the model group to which the model belongs,
        if available

        :return: ID of the model group for the model
        """
        return self._model_group_id

    @model_group_id.setter
    def model_group_id(self, id_: str):
        """
        Set the model group id for this model

        :param id: ID to set
        """
        self._model_group_id = id_

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the dictionary representation of the model

        :return:
        """
        return attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)

    @property
    def overview(self) -> str:
        """
        Returns a string that represents an overview of the model

        :return:
        """
        return pformat(self.to_dict())


@attr.s(auto_attribs=True)
class OptimizedModel(BaseModel):
    """
    Class representing an OptimizedModel in SC
    """
    model_status: str = attr.ib(
        kw_only=True, converter=str_to_enum_converter(ModelStatus)
    )
    optimization_methods: List[str] = attr.ib(kw_only=True)
    optimization_objectives: Dict[str, Any] = attr.ib(kw_only=True)
    optimization_type: str = attr.ib(
        kw_only=True, converter=str_to_enum_converter(OptimizationType)
    )


@attr.s(auto_attribs=True)
class Model(BaseModel):
    """
    Class representing a Model in SC
    """
    architecture: str = attr.ib(kw_only=True)
    version: int = attr.ib(kw_only=True)
    score_up_to_date: bool = attr.ib(kw_only=True)
    optimization_capabilities: OptimizationCapabilities = attr.ib(kw_only=True)
    optimized_models: List[OptimizedModel] = attr.ib(kw_only=True)

    @property
    def model_group_id(self) -> Optional[str]:
        """
        Returns the unique database ID of the model group to which the model belongs,
        if available

        :return: ID of the model group for the model
        """
        return self._model_group_id

    @model_group_id.setter
    def model_group_id(self, id_: str):
        """
        Set the model group id for this model

        :param id: ID to set
        """
        self._model_group_id = id_
        for model in self.optimized_models:
            model.model_group_id = id_
