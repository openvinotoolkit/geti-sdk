from sc_api_tools.data_models import ProjectStatus, Project
from sc_api_tools.http_session import SCSession
from sc_api_tools.rest_converters import StatusRESTConverter
from sc_api_tools.utils import get_supported_algorithms


class TrainingManager:
    """
    Class to manage training jobs for a certain project
    """

    def __init__(self, workspace_id: str, project: Project, session: SCSession):
        self.session = session
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/projects/{project.id}"
        self.supported_algos = get_supported_algorithms(session)

    def get_status(self) -> ProjectStatus:
        """
        Gets the current status of the project from the SC cluster
        :return:
        """
        response = self.session.get_rest_response(
            url=f"{self.base_url}/status",
            method="GET"
        )
        return StatusRESTConverter.from_dict(response)
