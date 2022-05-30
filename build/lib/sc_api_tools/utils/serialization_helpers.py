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
