import attr

from sc_api_tools.data_models.enums import AnnotationState
from sc_api_tools.data_models.utils import str_to_enum_converter_by_name_or_value


@attr.s(auto_attribs=True)
class TaskAnnotationState:
    """
    This class represents the state of an annotation for a particular task in the SC
    project
    """
    task_id: str
    state: str = attr.ib(
        converter=str_to_enum_converter_by_name_or_value(AnnotationState)
    )
