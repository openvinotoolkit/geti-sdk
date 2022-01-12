import attr

from sc_api_tools.data_models.utils import str_to_enum_converter
from sc_api_tools.data_models.enums import Domain


@attr.s(auto_attribs=True)
class Algorithm:
    """
    Class representing a supported algorithm in SC
    """
    algorithm_name: str
    domain: str = attr.ib(converter=str_to_enum_converter(Domain))
    model_size: str
    model_template_id: str
    gigaflops: float
