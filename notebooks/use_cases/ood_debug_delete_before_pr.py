import cv2

from geti_sdk import Geti
from geti_sdk.detect_ood.ood_model import COODModel
from geti_sdk.detect_ood.utils import get_usable_deployment
from geti_sdk.post_inference_hooks import (
    FileSystemDataCollection,
    OODTrigger,
    PostInferenceHook,
)
from geti_sdk.rest_clients import ModelClient, ProjectClient
from geti_sdk.utils import get_server_details_from_env

geti_server_configuration = get_server_details_from_env(
    env_file_path="/Users/rgangire/workspace/code/repos/Geti-SDK/dev/geti-sdk/notebooks/use_cases/.env"
)

geti = Geti(server_config=geti_server_configuration)
project_client = ProjectClient(session=geti.session, workspace_id=geti.workspace_id)

PROJECT_NAME = "CUB3"
project = project_client.get_project_by_name(project_name=PROJECT_NAME)

model_client = ModelClient(
    session=geti.session,
    workspace_id=geti.workspace_id,
    project=project,
)

deployment = get_usable_deployment(geti=geti, model_client=model_client)


ood_model = COODModel(geti=geti, project=project, deployment=deployment)


trigger = OODTrigger(
    ood_model=ood_model,
)

action = FileSystemDataCollection(
    target_folder="/Users/rgangire/workspace/Results/SDK/data/CollectedImages"
)

geti_hook = PostInferenceHook(
    trigger=trigger,
    action=action,
    max_threads=5,
    limit_action_rate=True,
    max_frames_per_second=1,
)

ood_model.deployment.add_post_inference_hook(hook=geti_hook)
dummy_imgae_path = "/Users/rgangire/workspace/Results/SDK/data/ood_images/Black_And_White_Warbler_0001_160352_669e4d1f62ebb5f7b69f97ad.jpg"

img = cv2.imread(dummy_imgae_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

prediction = ood_model.deployment.explain(
    image=img
)  # Set this the other way - give a deployment which xai to the ood_model or use ood_model's function
