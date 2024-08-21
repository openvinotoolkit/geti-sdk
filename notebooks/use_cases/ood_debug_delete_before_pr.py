# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.


# This is a script just to test the OOD trigger. This will be deleted after testing.
# A notebook will be made to showcase the usage of OOD trigger.

import cv2

from geti_sdk import Geti
from geti_sdk.detect_ood.ood_model import COODModel
from geti_sdk.detect_ood.utils import get_deployment_with_xai_head
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

PROJECT_NAME = "CUB6"
project = project_client.get_project_by_name(project_name=PROJECT_NAME)

model_client = ModelClient(
    session=geti.session,
    workspace_id=geti.workspace_id,
    project=project,
)

deployment = get_deployment_with_xai_head(geti=geti, model_client=model_client)

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
)

ood_model.deployment.add_post_inference_hook(hook=geti_hook)
# dummy_imgae_path = "/Users/rgangire/workspace/Results/SDK/data/ood_images/Black_And_White_Warbler_0001_160352_669e4d1f62ebb5f7b69f97ad.jpg"
# dummy_imgae_path = "/Users/rgangire/workspace/Results/SDK/data/images/Black_And_White_Warbler_0001_160352_669e4d1f62ebb5f7b69f97ad.jpg"
dummy_imgae_path = "/Users/rgangire/workspace/data/CUB_200_2011/CUB_200_2011/images/010.Red_winged_Blackbird/Red_Winged_Blackbird_0001_3695.jpg"
img = cv2.imread(dummy_imgae_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

prediction = ood_model.deployment.explain(image=img)
