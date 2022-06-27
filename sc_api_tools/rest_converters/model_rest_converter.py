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

from typing import Dict, Any

from sc_api_tools.data_models import ModelGroup, Model, OptimizedModel
from sc_api_tools.utils import deserialize_dictionary


class ModelRESTConverter:
    """
    Class that handles conversion of SC REST output for media entities to objects and
    vice versa.
    """

    @staticmethod
    def model_group_from_dict(input_dict: Dict[str, Any]) -> ModelGroup:
        """
        Converts a dictionary representing a model group to a ModelGroup object

        :param input_dict: Dictionary representing a model group, as returned by the
            SC /model_groups REST endpoint
        :return: ModelGroup object corresponding to the data in `input_dict`
        """
        return deserialize_dictionary(input_dict, output_type=ModelGroup)

    @staticmethod
    def model_from_dict(input_dict: Dict[str, Any]) -> Model:
        """
        Converts a dictionary representing a model to a Model object

        :param input_dict: Dictionary representing a model, as returned by the
            SC /model_groups/models REST endpoint
        :return: Model object corresponding to the data in `input_dict`
        """
        return deserialize_dictionary(input_dict, output_type=Model)

    @staticmethod
    def optimized_model_from_dict(input_dict: Dict[str, Any]) -> OptimizedModel:
        """
        Converts a dictionary representing an optimized model to a OptimizedModel object

        :param input_dict: Dictionary representing an optimized model, as returned by
            the SC /model_groups/models REST endpoint
        :return: OptimizedModel object corresponding to the data in `input_dict`
        """
        return deserialize_dictionary(input_dict, output_type=OptimizedModel)
