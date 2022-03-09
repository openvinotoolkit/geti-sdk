from typing import Optional

import attr

from sc_api_tools.data_models.utils import str_to_optional_enum_converter
from sc_api_tools.data_models.enums import Domain, TaskType


@attr.s(auto_attribs=True)
class Algorithm:
    """
    Class representing a supported algorithm in SC
    """
    algorithm_name: str
    
    model_size: str
    model_template_id: str
    gigaflops: float
    summary: Optional[str] = None
    domain: Optional[str] = attr.ib(  # `domain` is deprecated in SC1.1, replaced by task_type
        default=None, converter=str_to_optional_enum_converter(Domain)
    )
    task_type: Optional[str] = attr.ib(
        default=None, converter=str_to_optional_enum_converter(TaskType)
    )

    def __attrs_post_init__(self):
        """
        Convert domain to task type for backward compatibility with SC MVP
        
        :return: 
        """
        if self.domain is not None and self.task_type is None:
            self.task_type = TaskType.from_domain(self.domain)
            self.domain = None
