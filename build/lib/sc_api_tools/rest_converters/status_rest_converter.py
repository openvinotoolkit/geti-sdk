from typing import Dict, Any

from sc_api_tools.data_models import ProjectStatus
from sc_api_tools.utils import deserialize_dictionary


class StatusRESTConverter:
    """
    Class that handles conversion of SC REST output for status entities to objects
    """

    @staticmethod
    def from_dict(project_status_dict: Dict[str, Any]) -> ProjectStatus:
        """
        Creates a ProjectStatus instance from the input dictionary passed in
        `project_status_dict`

        :param project_status_dict: Dictionary representing the status of a project on
            the SC cluster
        :return: ProjectStatus instance, holding the status data contained in
            project_status_dict
        """
        return deserialize_dictionary(project_status_dict, output_type=ProjectStatus)
