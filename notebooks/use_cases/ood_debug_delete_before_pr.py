from geti_sdk import Geti
from geti_sdk.detect_ood.ood_model import COODModel
from geti_sdk.rest_clients import ProjectClient
from geti_sdk.utils import get_server_details_from_env

geti_server_configuration = get_server_details_from_env(
    env_file_path="/Users/rgangire/workspace/code/repos/Geti-SDK/dev/geti-sdk/notebooks/use_cases/.env"
)

geti = Geti(server_config=geti_server_configuration)
project_client = ProjectClient(session=geti.session, workspace_id=geti.workspace_id)

PROJECT_NAME = "CUB3"
project = project_client.get_project_by_name(project_name=PROJECT_NAME)

ood_model = COODModel(geti=geti, project=project)

a = 1
