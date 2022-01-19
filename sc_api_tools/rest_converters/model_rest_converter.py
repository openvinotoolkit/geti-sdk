from typing import Dict, Any

from sc_api_tools.data_models import ModelGroup, Model
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
