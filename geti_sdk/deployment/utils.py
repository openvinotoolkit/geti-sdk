# Copyright (C) 2022 Intel Corporation
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
import re
from importlib import resources
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from pathvalidate import sanitize_filepath

from geti_sdk.data_models import (
    Annotation,
    Label,
    OptimizedModel,
    Prediction,
    Project,
    ScoredLabel,
    Task,
    TaskType,
)
from geti_sdk.data_models.shapes import Rectangle
from geti_sdk.deployment.data_models import ROI, IntermediateInferenceResult

try:
    OVMS_README_PATH = str(
        resources.files("geti_sdk.deployment.resources") / "OVMS_README.md"
    )
except AttributeError:
    with resources.path("geti_sdk.deployment.resources", "OVMS_README.md") as data_path:
        OVMS_README_PATH = str(data_path)


def generate_ovms_model_name(
    project: Project, model: OptimizedModel, omit_version: bool = True
) -> str:
    """
    Generate a valid name for a model to be uploaded to the OpenVINO Model Server.

    :param project: Project the model belongs to
    :param model: The model for which to generate the name
    :param omit_version: True not to include the version of the model in the name.
        This is useful if the name needs to be used to create a directory structure,
        since including the version will break this.
    :return: String containing the name of the model
    """
    model_name = sanitize_filepath(project.name + "_" + model.name)
    model_name = model_name.replace(" ", "_")
    model_name = model_name.replace("-", "_")
    model_name = model_name.lower()
    if model.version is not None and not omit_version:
        model_name += f":{model.version}"
    return model_name


def generate_ovms_model_address(ovms_address: str, model_name: str) -> str:
    """
    Generate a string representing a valid model address for the OpenVINO Model Server,
    from a given `ovms_address` and `model_name`.

    The format expected for model addresses by OVMS is:

        <address>:<port>/models/<model_name>[:<model_version>]

    Where `address` must not contain the protocol (http or https)

    :param ovms_address: IP address or URL pointing to the OVMS instance, including
        port specification
    :param model_name: Name of the model
    :return: String containing the model address
    """
    model_address = f"{ovms_address}/models/{model_name}"
    if model_address.startswith("https://"):
        model_address = model_address[8:]
    if model_address.startswith("http://"):
        model_address = model_address[7:]
    return model_address


def target_device_is_ovms(device: str) -> bool:
    """
    Return True if the target `device` specified is a URL or IP address, False otherwise

    :param device: Target device string to check
    :return: True if the string represents a URL or IP address, False otherwise
    """
    # Check if 'device' has been specified as a URL or IP address.
    server_pattern = re.compile(
        r"^((https?://)|(www.))(?:([a-zA-Z]+)|(\d+\.\d+\.\d+\.\d+)):\d{1,5}?$"
    )
    return server_pattern.match(device) is not None


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert an RGB color value to its corresponding hexadecimal representation.

    :param rgb: A tuple representing the RGB color value, where each element is an integer between 0 and 255.
    :return: The hexadecimal representation of the RGB color value.

    _Example:

        >>> rgb_to_hex((255, 0, 0))
        '#ff0000'
    """
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def get_package_version_from_requirements(
    requirements_path: os.PathLike, package_name: str
) -> str:
    """
    Get the version of a package from a requirements file.

    :param requirements_path: The requirements file path
    :param package_name: The name of the package to get the version of
    :return: The version of the package as a string, empty line if the package is not found.
    :raises: ValueError If the requirements file is not found.
    """
    if not package_name:
        return ""

    requirements_path = Path(requirements_path).resolve()
    if not requirements_path.exists():
        raise ValueError(f"Requirements file {requirements_path} not found")

    for line in requirements_path.read_text().split("\n"):
        # The otx may be installed from a GitHub url
        if line.startswith(package_name) and "==" in line:
            return line.split("==")[1].strip()
    return ""


def assign_empty_label(
    prediction: Prediction, image: np.ndarray, empty_label: Optional[Label] = None
):
    """
    Modify the `prediction` in place to assign an empty label, if the prediction
    does not contain any objects.

    :param prediction: Prediction object to assign the empty label to, if needed
    :param image: The original image for which the prediction was generated
    :param empty_label: The Label object representing the empty label to assign
    """
    # Empty label is not generated by prediction postprocessing correctly, append
    # it here if there are no other predictions
    height, width = image.shape[0:2]
    if len(prediction.annotations) == 0:
        if empty_label is not None:
            prediction.append(
                Annotation(
                    shape=Rectangle(x=0, y=0, width=width, height=height),
                    labels=[ScoredLabel.from_label(empty_label, probability=1)],
                )
            )


def flow_control(
    intermediate_result: IntermediateInferenceResult, task: Task
) -> Tuple[List[ROI], List[np.ndarray]]:
    """
    Logic for the flow control tasks

    :param intermediate_result: IntermediateInferenceResult object, containing the
        predictions so far, as well as the rois.
    :param task: The task to perform the flow control operation for
    :return: Tuple containing:
        - rois -> A list of rois generated by the flow control task, to be passed
            down to the next task in the task chain
        - image_views -> A list of numpy arrays containing the image data in the
            corresponding roi
    """
    # CROP task
    if task.type == TaskType.CROP:
        rois = intermediate_result.filter_rois(label=None)
        image_views = intermediate_result.generate_views(rois)
    else:
        raise NotImplementedError(
            f"Unable to run inference for the pipeline in the deployed "
            f"project: Unsupported task type {task.type} found."
        )
    return rois, image_views


def pipeline_callback_factory(
    is_first_task: bool,
    is_final_task: bool,
    add_flow_control: bool,
    current_task: Task,
    deployment,
    final_callback: Callable[[Prediction, Optional[Any]], None],
    next_trainable_task: Optional[Task] = None,
    next_task: Optional[Task] = None,
) -> Callable[[Prediction, Any], None]:
    """
    Generate callback functions for pipeline inference
    """
    if is_first_task:

        def callback(result: Prediction, runtime_data: Tuple[np.ndarray, Any]):
            """
            Process inference results for the first task in a task chain, this
            initializes the intermediate result
            """
            image, runtime_user_data = runtime_data
            rois = [
                ROI.from_annotation(annotation) for annotation in result.annotations
            ]

            if len(rois) == 0:
                # Empty prediction. Assign empty label and run final callback immediately
                assign_empty_label(result, image, deployment._empty_label)
                final_callback(result, runtime_user_data)
                return

            intermediate_result = IntermediateInferenceResult(
                image=image, prediction=result, rois=rois
            )

            if add_flow_control:
                rois, image_views = flow_control(intermediate_result, next_task)
                intermediate_result.rois = rois
            else:
                image_views = intermediate_result.generate_views()

            for roi, view in zip(rois, image_views):
                deployment._infer_task_async(
                    image=view,
                    explain=False,
                    task=next_trainable_task,
                    runtime_data=(image, roi, intermediate_result, runtime_user_data),
                )

    else:

        def callback(
            result: Prediction,
            runtime_data: Tuple[np.ndarray, ROI, IntermediateInferenceResult, Any],
        ):
            """
            Process inference results for subsequent tasks
            """
            image, roi, intermediate_result, runtime_user_data = runtime_data

            if current_task.is_global:
                # Global tasks add their labels to the existing shape in the ROI
                intermediate_result.extend_annotations(result.annotations, roi=roi)
            else:
                # Local tasks create new shapes in the image coordinate system,
                # and generate ROI's corresponding to the new shapes
                for annotation in result.annotations:
                    intermediate_result.append_annotation(annotation, roi=roi)
                    new_roi = ROI.from_annotation(annotation)
                    intermediate_result.add_to_infer_queue(
                        new_roi.to_absolute_coordinates(parent_roi=roi)
                    )

            intermediate_result.increment_infer_counter()

            if intermediate_result.all_rois_inferred():
                # In this case, all ROIS from the preceding task have been inferred.
                # The infer queue now becomes the new ROIs for the intermediate result
                new_rois = intermediate_result.get_infer_queue()
                intermediate_result.reset_infer_counter()
                intermediate_result.rois = new_rois

                if add_flow_control:
                    rois, image_views = flow_control(intermediate_result, next_task)
                    intermediate_result.rois = rois
                else:
                    image_views = intermediate_result.generate_views()

                if not is_final_task:
                    for roi, view in zip(rois, image_views):
                        deployment._infer_task_async(
                            image=view,
                            explain=False,
                            task=next_trainable_task,
                            runtime_data=(
                                image,
                                roi,
                                intermediate_result,
                                runtime_user_data,
                            ),
                        )
                else:
                    prediction = intermediate_result.prediction
                    final_callback(prediction, runtime_user_data)

    return callback
