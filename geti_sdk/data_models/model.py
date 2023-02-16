# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import copy
import json
from pprint import pformat
from typing import Any, ClassVar, Dict, List, Optional

import attr

from geti_sdk.data_models.enums import ModelStatus, OptimizationType
from geti_sdk.data_models.utils import (
    attr_value_serializer,
    deidentify,
    remove_null_fields,
    str_to_datetime,
    str_to_enum_converter,
)
from geti_sdk.utils.serialization_helpers import deserialize_dictionary

from .label import Label
from .performance import Performance


@attr.define
class OptimizationCapabilities:
    """
    Representation of the various model optimization capabilities in GETi.
    """

    is_nncf_supported: bool
    is_filter_pruning_enabled: Optional[bool] = None  # deprecated in v1.1
    is_filter_pruning_supported: Optional[bool] = None


@attr.define(slots=False)
class BaseModel:
    """
    Representation of the basic information for a Model or OptimizedModel in GETi
    """

    _identifier_fields: ClassVar[str] = [
        "id",
        "previous_revision_id",
        "previous_trained_revision_id",
    ]

    name: str
    fps_throughput: str
    latency: str
    precision: List[str]
    creation_date: str = attr.field(converter=str_to_datetime)
    size: Optional[int] = None
    target_device: Optional[str] = None
    target_device_type: Optional[str] = None
    previous_revision_id: Optional[str] = None
    previous_trained_revision_id: Optional[str] = None
    score: Optional[float] = attr.field(default=None)  # 'score' is removed in v1.1
    performance: Optional[Performance] = None
    id: Optional[str] = attr.field(default=None)
    label_schema_in_sync: Optional[bool] = attr.field(default=None)  # Added in Geti 1.1

    def __attrs_post_init__(self):
        """
        Initialize private attributes.
        """
        self._model_group_id: Optional[str] = None
        self._base_url: Optional[str] = None

    @property
    def model_group_id(self) -> Optional[str]:
        """
        Return the unique database ID of the model group to which the model belongs,
        if available.

        :return: ID of the model group for the model
        """
        return self._model_group_id

    @model_group_id.setter
    def model_group_id(self, id_: str):
        """
        Set the model group id for this model.

        :param id_: ID to set
        """
        self._model_group_id = id_

    @property
    def base_url(self) -> Optional[str]:
        """
        Return the base url that can be used to get the model details, download the
        model, etc., if available.

        :return: base url at which the model can be addressed. The url is defined
            relative to the ip address or hostname of the Intel® Geti™ server
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
        Set the base url that can be used to get the model details, download the
        model, etc.

        :param base_url: base url at which the model can be addressed
        :return:
        """
        if self.model_group_id is not None:
            if self.model_group_id in base_url:
                if base_url.endswith(f"models/{self.id}"):
                    base_url = base_url
                else:
                    base_url += f"/models/{self.id}"
            else:
                base_url += f"/{self.model_group_id}/models/{self.id}"
        else:
            base_url = base_url
        if hasattr(self, "optimized_models"):
            for model in self.optimized_models:
                model._base_url = base_url + f"/optimized_models/{model.id}"
        self._base_url = base_url

    @model_group_id.setter
    def model_group_id(self, id_: str):
        """
        Set the model group id for this model.

        :param id: ID to set
        """
        self._model_group_id = id_

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the dictionary representation of the model.

        :return:
        """
        return attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)

    @property
    def overview(self) -> str:
        """
        Return a string that represents an overview of the model.

        :return:
        """
        deidentified = copy.deepcopy(self)
        deidentified.deidentify()
        overview_dict = deidentified.to_dict()
        remove_null_fields(overview_dict)
        return pformat(overview_dict)

    def deidentify(self) -> None:
        """
        Remove unique database IDs from the BaseModel.
        """
        deidentify(self)


@attr.define(slots=False)
class OptimizedModel(BaseModel):
    """
    Representation of an OptimizedModel in Intel® Geti™. An optimized model is a trained model
    that has been converted OpenVINO representation. This conversion may involve weight
    quantization, filter pruning, or other optimization techniques supported by
    OpenVINO.
    """

    model_status: str = attr.field(
        kw_only=True, converter=str_to_enum_converter(ModelStatus)
    )
    optimization_methods: List[str] = attr.field(kw_only=True)
    optimization_objectives: Dict[str, Any] = attr.field(kw_only=True)
    optimization_type: str = attr.field(
        kw_only=True, converter=str_to_enum_converter(OptimizationType)
    )
    version: Optional[int] = attr.field(kw_only=True, default=None)


@attr.define(slots=False)
class Model(BaseModel):
    """
    Representation of a trained Model in Intel® Geti™.
    """

    architecture: str = attr.field(kw_only=True)
    score_up_to_date: bool = attr.field(kw_only=True)
    optimization_capabilities: OptimizationCapabilities = attr.field(kw_only=True)
    optimized_models: List[OptimizedModel] = attr.field(kw_only=True)
    labels: Optional[List[Label]] = None
    version: Optional[int] = attr.field(default=None, kw_only=True)
    # 'version' is deprecated in v1.1
    training_dataset_info: Optional[Dict[str, str]] = None

    @property
    def model_group_id(self) -> Optional[str]:
        """
        Return the unique database ID of the model group to which the model belongs,
        if available.

        :return: ID of the model group for the model
        """
        return self._model_group_id

    @model_group_id.setter
    def model_group_id(self, id_: str):
        """
        Set the model group id for this model.

        :param id: ID to set
        """
        self._model_group_id = id_
        for model in self.optimized_models:
            model.model_group_id = id_

    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any]) -> "Model":
        """
        Create a Model instance from a dictionary holding the model data.

        :param model_dict: Dictionary representing a model
        :return: Model instance reflecting the data contained in `model_dict`
        """
        return deserialize_dictionary(model_dict, cls)

    @classmethod
    def from_file(cls, filepath: str) -> "Model":
        """
        Create a Model instance from a .json file holding the model data.

        :param filepath: Path to a json file holding the model data
        :return:
        """
        with open(filepath, "r") as file:
            model_dict = json.load(file)
        return cls.from_dict(model_dict=model_dict)
