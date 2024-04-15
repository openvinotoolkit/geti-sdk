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

from pprint import pformat
from typing import Any, Dict, Optional

import attr

from geti_sdk.data_models.enums import Domain, TaskType
from geti_sdk.data_models.utils import str_to_optional_enum_converter

from .utils import attr_value_serializer, remove_null_fields


@attr.define
class Algorithm:
    """
    Representation of a supported algorithm on the Intel® Geti™ platform.
    """

    model_size: str
    model_template_id: str
    gigaflops: float
    algorithm_name: Optional[str] = None  # Deprecated in Geti v1.16, use 'name' instead
    name: Optional[str] = None
    summary: Optional[str] = None
    domain: Optional[str] = attr.field(
        default=None, converter=str_to_optional_enum_converter(Domain)
    )
    # `domain` is deprecated in SC1.1, replaced by task_type
    task_type: Optional[str] = attr.field(
        default=None, converter=str_to_optional_enum_converter(TaskType)
    )
    supports_auto_hpo: Optional[bool] = None
    recommended_choice: Optional[bool] = (
        None  # Deprecated in Geti v1.16, use 'default_algorithm' instead
    )
    default_algorithm: Optional[bool] = None  # Added in Geti v1.16
    performance_category: Optional[str] = None  # Added in Geti v1.9
    lifecycle_stage: Optional[str] = None  # Added in Geti v1.9

    def __attrs_post_init__(self):
        """
        Convert domain to task type for backward compatibility with earlier versions of
        the Intel® Geti™ platform
        """
        if self.default_algorithm is None:
            # For older Geti versions, that were still using 'recommended choice'
            self.default_algorithm = self.recommended_choice
        if self.name is None:
            # For older Geti versions, that were still using 'algorithm_name'
            self.name = self.algorithm_name
        if self.domain is not None and self.task_type is None:
            self.task_type = TaskType.from_domain(self.domain)
            self.domain = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Algorithm to a dictionary representation.

        :return: Dictionary holding the algorithm data
        """
        output_dict = attr.asdict(self, value_serializer=attr_value_serializer)
        return output_dict

    @property
    def overview(self) -> str:
        """
        Return a string that shows an overview of the Algorithm properties.

        :return: String holding an overview of the algorithm
        """
        overview_dict = self.to_dict()
        remove_null_fields(overview_dict)
        return pformat(overview_dict)
