from typing import Dict, Any, Type, cast

from omegaconf import OmegaConf

from sc_api_tools.data_models.media_list import MediaTypeVar


class MediaRESTConverter:
    """
    Class that handles conversion of SC REST output for media entities to objects and
    vice versa.
    """

    @staticmethod
    def from_dict(
            input_dict: Dict[str, Any],
            media_type: Type[MediaTypeVar]
    ) -> MediaTypeVar:
        """
        Creates an instance of type `media_type` representing a media entity in SC
        from a dictionary returned by the SC /media REST endpoints

        :param input_dict: Dictionary representing the media entity
        :param media_type: Type of the media entity
        :return: Instance of type `media_type` containing the entity represented in
            the REST input in `input_dict`.
        """
        media_dict_config = OmegaConf.create(input_dict)
        schema = OmegaConf.structured(media_type)
        config = OmegaConf.merge(schema, media_dict_config)
        return cast(media_type, OmegaConf.to_object(config))
