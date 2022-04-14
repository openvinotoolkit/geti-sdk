import copy
import json
from pprint import pformat
from typing import Optional, List, Dict, Any, ClassVar

import attr

from sc_api_tools.data_models.enums import ModelStatus, OptimizationType
from sc_api_tools.data_models.utils import (
    str_to_datetime,
    str_to_enum_converter,
    attr_value_serializer,
    deidentify
)
from sc_api_tools.utils import deserialize_dictionary
from sc_api_tools.utils.dictionary_helpers import remove_null_fields

from .performance import Performance


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
    _identifier_fields: ClassVar[str] = [
        "id", "previous_revision_id", "previous_trained_revision_id"
    ]

    name: str
    fps_throughput: str
    latency: str
    precision: List[str]
    creation_date: str = attr.ib(converter=str_to_datetime)
    size: Optional[int] = None
    target_device: Optional[str] = None
    target_device_type: Optional[str] = None
    previous_revision_id: Optional[str] = None
    previous_trained_revision_id: Optional[str] = None
    score: Optional[float] = attr.ib(default=None)  # 'score' is removed in v1.1
    performance: Optional[Performance] = None
    id: Optional[str] = attr.ib(default=None)

    def __attrs_post_init__(self):
        self._model_group_id: Optional[str] = None
        self._base_url: Optional[str] = None

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

        :param id_: ID to set
        """
        self._model_group_id = id_

    @property
    def base_url(self) -> Optional[str]:
        """
        Returns the base url that can be used to get the model details, download the
        model, etc., if available

        :return: base url at which the model can be addressed. The url is defined
            relative to the ip address or hostname of the SC cluster
        """
        if self._base_url is not None:
            return self._base_url
        else:
            raise ValueError(
                f"Insufficient data to determine base url for model {self}. Please "
                f"make sure that property `base_url` is set first."
            )

    @base_url.setter
    def base_url(self, base_url: str):
        """
        Sets the base url that can be used to get the model details, download the
        model, etc.

        :param base_url: base url at which the model can be addressed
        :return:
        """
        if self.model_group_id is not None:
            if self.model_group_id in base_url:
                if base_url.endswith(f'models/{self.id}'):
                    base_url = base_url
                else:
                    base_url += f'/models/{self.id}'
            else:
                base_url += f'/{self.model_group_id}/models/{self.id}'
        else:
            base_url = base_url
        if hasattr(self, 'optimized_models'):
            for model in self.optimized_models:
                model._base_url = base_url + f'/optimized_models/{model.id}'
        self._base_url = base_url

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
        deidentified = copy.deepcopy(self)
        deidentified.deidentify()
        overview_dict = deidentified.to_dict()
        remove_null_fields(overview_dict)
        return pformat(overview_dict)

    def deidentify(self):
        """
        Removes unique database IDs from the BaseModel
        :return:
        """
        deidentify(self)


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
    score_up_to_date: bool = attr.ib(kw_only=True)
    optimization_capabilities: OptimizationCapabilities = attr.ib(kw_only=True)
    optimized_models: List[OptimizedModel] = attr.ib(kw_only=True)
    version: Optional[int] = attr.ib(default=None, kw_only=True)  # 'version' is deprecated in v1.1

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

    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any]) -> 'Model':
        """
        Creates a Model instance from a dictionary holding the model data

        :param model_dict: Dictionary representing a model
        :return: Model instance reflecting the data contained in `model_dict`
        """
        return deserialize_dictionary(model_dict, cls)

    @classmethod
    def from_file(cls, filepath: str) -> 'Model':
        """
        Creates a Model instance from a .json file holding the model data

        :param filepath: Path to a json file holding the model data
        :return:
        """
        with open(filepath, 'r') as file:
            model_dict = json.load(file)
        return cls.from_dict(model_dict=model_dict)
