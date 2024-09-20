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
import os
import shutil

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

# This is a script just to test the OOD trigger. This will be deleted after testing.
# A notebook will be made to showcase the usage of OOD trigger.


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

ood_model = COODModel(
    geti=geti,
    project=project,
    deployment=deployment,
    ood_images_dir="/Users/rgangire/workspace/Results/SDK/data/ood_near",
    workspace_dir="/Users/rgangire/workspace/Results/SDK/COOD_MODEL_WS",
)

trigger = OODTrigger(ood_model=ood_model, threshold=ood_model.best_thresholds["fscore"])

action = FileSystemDataCollection(
    target_folder="/Users/rgangire/workspace/Results/SDK/data/CollectedImages-NEW",
    # file_name_prefix="OOD",
    save_predictions=False,
    save_scores=False,
    save_overlays=False,
)

geti_hook = PostInferenceHook(
    trigger=trigger,
    action=action,
)

ood_model.deployment.add_post_inference_hook(hook=geti_hook)
# ood_dir = "/Users/rgangire/workspace/Results/SDK/data/TestOOD"
# for img_file in os.listdir(ood_dir):
#     img_path = os.path.join(ood_dir, img_file)
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     prediction = ood_model.deployment.explain(image=img)
#     probability = prediction.annotations[0].labels[0].probability
#     label = prediction.annotations[0].labels[0].name
#     print(f"Image: {img_file}, Label: {label}, Probability: {probability}")


# from geti_sdk.demos import EXAMPLE_IMAGE_PATH

# example_path = "/Users/rgangire/workspace/data/CUB_200_2011/CUB_200_2011/images/013.Bobolink/Bobolink_0099_9314.jpg"
# numpy_image = cv2.imread(EXAMPLE_IMAGE_PATH)
# numpy_rgb = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
#
# prediction = deployment.explain(numpy_rgb)
# print(
#     f"Predicted as : {prediction.get_label_names()[0]} with a probability: {100*prediction.annotations[0].labels[0].probability:.1f}%"
# )


OOD_BASE_PATH = "/Users/rgangire/workspace/Results/SDK/data/TestOOD_ALL-Extended-Pred90"
all_ood_collection = "/Users/rgangire/workspace/Results/SDK/data/CollectedImages-NEW"
prob_treatment = False

ood_sub_folders = os.listdir(OOD_BASE_PATH)
for sub_folder in ood_sub_folders:
    sub_folder_path = os.path.join(OOD_BASE_PATH, sub_folder)
    if not os.path.isdir(sub_folder_path):
        continue
    # clean all the files in all_ood_collection
    ood_images_path = os.path.join(all_ood_collection, "images")
    shutil.rmtree(ood_images_path, ignore_errors=True)
    os.makedirs(ood_images_path, exist_ok=True)
    print(f"Processing OOD sub-folder: {sub_folder}")
    if prob_treatment:
        copy_path_pred_90 = os.path.join(all_ood_collection, sub_folder + "_pred_90")
        os.makedirs(copy_path_pred_90, exist_ok=True)

    count_all_images = 0
    count_pred_90 = 0
    for img_file in os.listdir(sub_folder_path):
        img_path = os.path.join(sub_folder_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prediction = ood_model.deployment.explain(image=img)
        probability = prediction.annotations[0].labels[0].probability
        count_all_images += 1
        if prob_treatment:
            if probability >= 0.9:
                shutil.copy(img_path, copy_path_pred_90)
                count_pred_90 += 1

    images_in_ood_path = os.listdir(ood_images_path)
    images_in_ood_path = [
        image for image in images_in_ood_path if image.endswith(".png")
    ]
    print(
        f"Total images: {count_all_images}, "
        f"Images with probability >= 0.9: {count_pred_90}, "
        f"Images Detected as OOD: {len(images_in_ood_path)}"
    )
