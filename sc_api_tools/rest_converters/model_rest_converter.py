from typing import Dict, Any, cast

from omegaconf import OmegaConf

from sc_api_tools.data_models import ModelGroup, Model


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
        model_group_dict_config = OmegaConf.create(input_dict)
        schema = OmegaConf.structured(ModelGroup)
        values = OmegaConf.merge(schema, model_group_dict_config)
        return cast(ModelGroup, OmegaConf.to_object(values))

    @staticmethod
    def model_from_dict(input_dict: Dict[str, Any]) -> Model:
        """
        Converts a dictionary representing a model to a Model object

        :param input_dict: Dictionary representing a model, as returned by the
            SC /model_groups/models REST endpoint
        :return: Model object corresponding to the data in `input_dict`
        """
        model_dict_config = OmegaConf.create(input_dict)
        schema = OmegaConf.structured(Model)
        values = OmegaConf.merge(schema, model_dict_config)
        return cast(Model, OmegaConf.to_object(values))