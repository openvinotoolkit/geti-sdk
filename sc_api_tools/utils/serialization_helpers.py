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

from typing import Dict, Any, cast, TypeVar, Type

from omegaconf import OmegaConf

OutputTypeVar = TypeVar("OutputTypeVar")


def deserialize_dictionary(
        input_dictionary: Dict[str, Any], output_type: Type[OutputTypeVar]
) -> OutputTypeVar:
    """
    Deserialize an `input_dictionary` to an object of the type passed in `output_type`

    :param input_dictionary: Dictionary to deserialize
    :param output_type: Type of the object that the dictionary represents, and to
        which the data will be deserialized
    :return: Object of type `output_type`, holding the data passed in
        `input_dictionary`.
    """
    model_dict_config = OmegaConf.create(input_dictionary)
    schema = OmegaConf.structured(output_type)
    values = OmegaConf.merge(schema, model_dict_config)
    return cast(output_type, OmegaConf.to_object(values))
