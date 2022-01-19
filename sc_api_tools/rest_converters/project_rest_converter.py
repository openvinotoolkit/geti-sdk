from typing import Dict, Any

import copy

from sc_api_tools.data_models import Project
from sc_api_tools.utils import deserialize_dictionary, remove_null_fields


class ProjectRESTConverter:
    """
    Class that handles conversion of SC REST output for project entities to objects and
    vice versa.
    """

    @classmethod
    def from_dict(cls, project_input: Dict[str, Any]) -> Project:
        """
        Creates a Project from a dictionary representing a project, as
        returned by the /projects endpoint in SC.

        :param project_input: Dictionary representing a project, as returned by SC
        :return: Project object representing the project given in `project_input`
        """
        prepared_project = copy.deepcopy(project_input)
        for connection in prepared_project["pipeline"]["connections"]:
            from_ = connection.pop("from", None)
            if from_ is not None:
                connection.update({"from_": from_})
        return deserialize_dictionary(prepared_project, output_type=Project)


    @classmethod
    def to_dict(cls, project: Project) -> Dict[str, Any]:
        """
        Converts the `project` to its dictionary representation.
        This functions removes database UID's and optional fields that are `None`
        from the output dictionary, to make the output more compact and improve
        readability.

        :param project: Project to convert to dictionary
        :return:
        """
        project.deidentify()
        project_data = project.to_dict()
        remove_null_fields(project_data)
        return project_data
