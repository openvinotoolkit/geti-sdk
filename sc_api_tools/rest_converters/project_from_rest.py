from typing import Dict, Any, cast

import copy

from omegaconf import OmegaConf

from sc_api_tools.data_models import Project


class ProjectRESTConverter:

    @classmethod
    def from_dict(cls, project_input: Dict[str, Any]):
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
        project_dict_config = OmegaConf.create(prepared_project)
        schema = OmegaConf.structured(Project)
        config = OmegaConf.merge(schema, project_dict_config)
        return cast(Project, OmegaConf.to_object(config))
