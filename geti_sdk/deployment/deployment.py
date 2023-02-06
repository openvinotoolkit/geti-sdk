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

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Union

import attr
import numpy as np

from geti_sdk.data_models import (
    Annotation,
    Label,
    Prediction,
    Project,
    ScoredLabel,
    Task,
    TaskType,
)
from geti_sdk.data_models.shapes import Polygon, Rectangle, RotatedRectangle
from geti_sdk.deployment.data_models import ROI, IntermediateInferenceResult
from geti_sdk.rest_converters import ProjectRESTConverter

from .deployed_model import DeployedModel
from .utils import OVMS_README_PATH, OVMS_REQ_PATH, generate_ovms_model_name


@attr.define(slots=False)
class Deployment:
    """
    Representation of a deployed Intel® Geti™ project that can be used to run
    inference locally
    """

    project: Project
    models: List[DeployedModel]

    def __attrs_post_init__(self):
        """
        Initialize private attributes.
        """
        self._is_single_task: bool = len(self.project.get_trainable_tasks()) == 1
        self._are_models_loaded: bool = False
        self._inference_converters: Dict[str, Any] = {}
        self._alternate_inference_converters: Dict[str, Any] = {}
        self._empty_labels: Dict[str, Label] = {}
        self._path_to_temp_resources: Optional[str] = None
        self._requires_resource_cleanup: bool = False

    @property
    def is_single_task(self) -> bool:
        """
        Return True if the deployment represents a project with only a single task.

        :return: True if the deployed project contains only one trainable task, False
            if it is a pipeline project
        """
        return self._is_single_task

    @property
    def are_models_loaded(self) -> bool:
        """
        Return True if all inference models for the Deployment are loaded and ready
        to infer.

        :return: True if all inference models for the deployed project are loaded in
            memory and ready for inference
        """
        return self._are_models_loaded

    def save(self, path_to_folder: Union[str, os.PathLike]):
        """
        Save the Deployment instance to a folder on local disk.

        :param path_to_folder: Folder to save the deployment to
        """
        project_dict = ProjectRESTConverter.to_dict(self.project)
        deployment_folder = os.path.join(path_to_folder, "deployment")

        os.makedirs(deployment_folder, exist_ok=True, mode=0o770)
        # Save project data
        project_filepath = os.path.join(deployment_folder, "project.json")
        with open(project_filepath, "w") as project_file:
            json.dump(project_dict, project_file, indent=4)
        # Save model for each task
        for task, model in zip(self.project.get_trainable_tasks(), self.models):
            model_dir = os.path.join(deployment_folder, task.title)
            os.makedirs(model_dir, exist_ok=True, mode=0o770)
            model.save(model_dir)

        # Clean up temp resources if needed
        self._remove_temporary_resources()
        self._requires_resource_cleanup = False

    @classmethod
    def from_folder(cls, path_to_folder: Union[str, os.PathLike]) -> "Deployment":
        """
        Create a Deployment instance from a specified `path_to_folder`.

        :param path_to_folder: Path to the folder containing the Deployment data
        :return: Deployment instance corresponding to the deployment data in the folder
        """
        deployment_folder = path_to_folder
        if not path_to_folder.endswith("deployment"):
            if "deployment" in os.listdir(path_to_folder):
                deployment_folder = os.path.join(path_to_folder, "deployment")
            else:
                raise ValueError(
                    f"No `deployment` folder found in the directory at "
                    f"`{path_to_folder}`. Unable to load Deployment."
                )
        project_filepath = os.path.join(deployment_folder, "project.json")
        with open(project_filepath, "r") as project_file:
            project_dict = json.load(project_file)
        project = ProjectRESTConverter.from_dict(project_dict)
        task_folder_names = [task.title for task in project.get_trainable_tasks()]
        models: List[DeployedModel] = []
        for task_folder in task_folder_names:
            models.append(
                DeployedModel.from_folder(os.path.join(deployment_folder, task_folder))
            )
        return cls(models=models, project=project)

    def load_inference_models(self, device: str = "CPU"):
        """
        Load the inference models for the deployment to the specified device.

        :param device: Device to load the inference models to
        """
        try:
            from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import (
                DetectionBoxToAnnotationConverter,
                IPredictionToAnnotationConverter,
                create_converter,
            )
        except ImportError as error:
            raise ValueError(
                f"Unable to load inference model for {self}. Relevant OpenVINO "
                f"packages were not found. Please make sure that all packages from the "
                f"file `requirements-deployment.txt` have been installed. "
            ) from error

        inference_converters: Dict[str, IPredictionToAnnotationConverter] = {}
        empty_labels: Dict[str, Label] = {}
        for model, task in zip(self.models, self.project.get_trainable_tasks()):
            model.load_inference_model(device=device, project=self.project)

            # This is a workaround for a bug in the label schema for anomaly tasks
            if task.type.is_anomaly:
                # For some reason the `is_anomaly` flag is not set correctly in the
                # ote_label_schema, which will break loading the prediction converter.
                # We set the flag here
                for label in model.ote_label_schema.get_labels(include_empty=True):
                    if label.name == "Anomalous":
                        label.is_anomalous = True

            inference_converter = create_converter(
                converter_type=task.type.to_ote_domain(), labels=model.ote_label_schema
            )
            inference_converters.update({task.title: inference_converter})

            # This is a workaround for a backwards incompatible change in later ote
            # versions
            if task.type.is_detection:
                alternate_inference_converter = DetectionBoxToAnnotationConverter(
                    labels=model.ote_label_schema
                )
                self._alternate_inference_converters.update(
                    {task.title: alternate_inference_converter}
                )
            empty_label = next((label for label in task.labels if label.is_empty), None)
            empty_labels.update({task.title: empty_label})

        self._inference_converters = inference_converters
        self._empty_labels = empty_labels
        self._are_models_loaded = True
        logging.info("Inference models loaded on device `{device}` successfully.")

    def infer(self, image: np.ndarray) -> Prediction:
        """
        Run inference on an image for the full model chain in the deployment.

        :param image: Image to run inference on, as a numpy array containing the pixel
            data. The image is expected to have dimensions [height x width x channels],
            with the channels in RGB order
        :return: inference results
        """
        if not self.are_models_loaded:
            raise ValueError(
                f"Deployment '{self}' is not ready to infer, the inference models are "
                f"not loaded. Please call 'load_inference_models' first."
            )

        # Single task inference
        if self.is_single_task:
            return self._infer_task(image, task=self.project.get_trainable_tasks()[0])

        previous_labels: Optional[List[Label]] = None
        intermediate_result: Optional[IntermediateInferenceResult] = None
        rois: Optional[List[ROI]] = None
        image_views: Optional[List[np.ndarray]] = None

        # Pipeline inference
        for task in self.project.pipeline.tasks[1:]:

            # First task in the pipeline generates the initial result and ROIs
            if task.is_trainable and previous_labels is None:
                task_prediction = self._infer_task(image, task=task)
                rois: Optional[List[ROI]] = None
                if not task.is_global:
                    rois = [
                        ROI.from_annotation(annotation)
                        for annotation in task_prediction.annotations
                    ]
                intermediate_result = IntermediateInferenceResult(
                    image=image, prediction=task_prediction, rois=rois
                )
                previous_labels = [label for label in task.labels if not label.is_empty]

            # Downstream trainable tasks
            elif task.is_trainable:
                if rois is None or image_views is None or intermediate_result is None:
                    raise NotImplementedError(
                        "Unable to run inference for the pipeline in the deployed "
                        "project: A flow control task is required between each "
                        "trainable task in the pipeline."
                    )
                new_rois: List[ROI] = []
                for roi, view in zip(rois, image_views):
                    view_prediction = self._infer_task(view, task)
                    if task.is_global:
                        # Global tasks add their labels to the existing shape in the ROI
                        intermediate_result.extend_annotations(
                            view_prediction.annotations, roi=roi
                        )
                    else:
                        # Local tasks create new shapes in the image coordinate system,
                        # and generate ROI's corresponding to the new shapes
                        for annotation in view_prediction.annotations:
                            intermediate_result.append_annotation(annotation, roi=roi)
                            new_rois.append(ROI.from_annotation(annotation))
                        intermediate_result.rois = [
                            new_roi.to_absolute_coordinates(parent_roi=roi)
                            for new_roi in new_rois
                        ]
                previous_labels = [label for label in task.labels if not label.is_empty]

            # Downstream flow control tasks
            else:
                if previous_labels is None:
                    raise NotImplementedError(
                        f"Unable to run inference for the pipeline in the deployed "
                        f"project: First task in the pipeline after the DATASET task "
                        f"has to be a trainable task, found task of type {task.type} "
                        f"instead."
                    )
                # CROP task
                if task.type == TaskType.CROP:
                    rois = intermediate_result.filter_rois(label=None)
                    image_views = intermediate_result.generate_views(rois)
                else:
                    raise NotImplementedError(
                        f"Unable to run inference for the pipeline in the deployed "
                        f"project: Unsupported task type {task.type} found."
                    )
        return intermediate_result.prediction

    def _infer_task(self, image: np.ndarray, task: Task) -> Prediction:
        """
        Run pre-processing, inference, and post-processing on the input `image`, for
        the model associated with the `task`.

        :param image: Image to run inference on
        :param task: Task to run inference for
        :return: Inference result
        """
        model = self._get_model_for_task(task)
        preprocessed_image, metadata = model.preprocess(image)
        inference_results = model.infer(preprocessed_image)
        postprocessing_results = model.postprocess(inference_results, metadata=metadata)
        converter = self._inference_converters[task.title]

        width: int = image.shape[1]
        height: int = image.shape[0]

        try:
            n_outputs = len(postprocessing_results)
        except TypeError:
            n_outputs = 1

        if n_outputs != 0:
            # The try/except is a workaround to handle different detection inference
            # results by different ote sdk versions
            try:
                annotation_scene_entity = converter.convert_to_annotation(
                    predictions=postprocessing_results, metadata=metadata
                )
            except TypeError as error:
                if task.type.is_detection:
                    converter = self._alternate_inference_converters[task.title]
                    annotation_scene_entity = converter.convert_to_annotation(
                        predictions=postprocessing_results, metadata=metadata
                    )
                    # Make sure next time we get it right in one shot
                    self._inference_converters.update({task.title: converter})
                else:
                    raise error

            prediction = Prediction.from_ote(
                annotation_scene_entity, image_width=width, image_height=height
            )
        else:
            prediction = Prediction(annotations=[])

        # Empty label is not generated by OTE correctly, append it here if there are
        # no other predictions
        if len(prediction.annotations) == 0:
            if self._empty_labels[task.title] is not None:
                prediction.append(
                    Annotation(
                        shape=Rectangle(x=0, y=0, width=width, height=height),
                        labels=[
                            ScoredLabel.from_label(
                                self._empty_labels[task.title], probability=1
                            )
                        ],
                    )
                )

        # Rotated detection models produce Polygons, convert them here to
        # RotatedRectangles
        if task.type == TaskType.ROTATED_DETECTION:

            for annotation in prediction.annotations:
                if isinstance(annotation.shape, Polygon):
                    annotation.shape = RotatedRectangle.from_polygon(annotation.shape)
        return prediction

    def _get_model_for_task(self, task: Task) -> DeployedModel:
        """
        Get the DeployedModel instance corresponding to the input `task`.

        :param task: Task to get the model for
        :return: DeployedModel corresponding to the task
        """
        try:
            task_index = self.project.get_trainable_tasks().index(task)
        except ValueError as error:
            raise ValueError(
                f"Task {task.title} is not in the list of trainable tasks for project "
                f"{self.project.name}."
            ) from error
        return self.models[task_index]

    def _remove_temporary_resources(self) -> None:
        """
        If necessary, clean up any temporary resources associated with the deployment.
        """
        if os.path.isdir(self._path_to_temp_resources):
            shutil.rmtree(self._path_to_temp_resources)
        else:
            logging.debug(
                f"Unable to clean up temporary resources for deployment {self}, "
                f"because the resources were not found on the system. Possibly "
                f"they were already deleted."
            )

    def __del__(self):
        """
        If necessary, clean up any temporary resources associated with the deployment.
        This method is called when the Deployment instance is deleted.
        """
        if self._requires_resource_cleanup:
            self._remove_temporary_resources()

    def generate_ovms_config(self, output_folder: Union[str, os.PathLike]) -> None:
        """
        Generate the configuration files needed to push the models for the
        `Deployment` instance to OVMS.

        :param output_folder: Target folder to save the configuration files to
        """
        # First prepare the model config list
        if os.path.basename(output_folder) != "ovms_models":
            ovms_models_dir = os.path.join(output_folder, "ovms_models")
        else:
            ovms_models_dir = output_folder
            output_folder = os.path.dirname(ovms_models_dir)
        os.makedirs(ovms_models_dir, exist_ok=True)

        model_configs: List[Dict[str, Dict[str, Any]]] = []
        for model in self.models:
            # Create configuration entry for model
            model_name = generate_ovms_model_name(
                project=self.project, model=model, omit_version=True
            )
            config = {
                "name": model_name,
                "base_path": f"/models/{model_name}",
                "shape": "auto",
            }
            model_configs.append({"config": config})

            # Copy IR model files to the expected OVMS format
            if model.version is not None:
                model_version = str(model.version)
            else:
                # Fallback to version 1 if no version info is available
                model_version = "1"

            ovms_model_dir = os.path.join(ovms_models_dir, model_name, model_version)
            source_model_dir = model.model_data_path
            os.makedirs(ovms_model_dir, exist_ok=True)
            for model_file in os.listdir(source_model_dir):
                shutil.copy2(
                    src=os.path.join(source_model_dir, model_file),
                    dst=os.path.join(ovms_model_dir, model_file),
                )

        # Save model configurations
        ovms_config_list = {"model_config_list": model_configs}
        config_target_filepath = os.path.join(ovms_models_dir, "ovms_model_config.json")
        with open(config_target_filepath, "w") as file:
            json.dump(ovms_config_list, file)

        # Copy resource files
        shutil.copy2(OVMS_README_PATH, os.path.join(output_folder))
        shutil.copy2(OVMS_REQ_PATH, os.path.join(output_folder))

        logging.info(
            f"Configuration files for OVMS model deployment have been generated in "
            f"directory '{output_folder}'. This folder contains a `OVMS_README.md` "
            f"file with instructions on how to launch OVMS, connect to it and run "
            f"inference. Please follow the instructions outlined there to get started."
        )
