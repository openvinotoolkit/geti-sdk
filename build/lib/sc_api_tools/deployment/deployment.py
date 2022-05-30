import json
import os
from typing import List, Union, Optional, Dict, Any

import attr

import numpy as np

from ote_sdk.entities.label_schema import LabelSchemaEntity

from sc_api_tools.data_models import (
    Project,
    Task,
    TaskType,
    Prediction,
    Label, Annotation, ScoredLabel,
)
from sc_api_tools.data_models.shapes import Rectangle
from sc_api_tools.deployment.data_models import ROI, IntermediateInferenceResult
from sc_api_tools.deployment.deployed_model import DeployedModel
from sc_api_tools.rest_converters import ProjectRESTConverter


@attr.s(auto_attribs=True)
class Deployment:
    """
    This class represents a deployed SC project that can be used to run inference
    locally
    """
    project: Project
    models: List[DeployedModel]

    def __attrs_post_init__(self):
        """
        Initializes private attributes
        """
        self._is_single_task: bool = len(self.project.get_trainable_tasks()) == 1
        self._are_models_loaded: bool = False
        self._inference_converters: Dict[str, Any] = {}
        self._empty_labels: Dict[str, Label] = {}

    @property
    def is_single_task(self) -> bool:
        """
        Returns True if the deployment represents a project with only a single task

        :return: True if the deployed project contains only one trainable task, False
            if it is a pipeline project
        """
        return self._is_single_task

    @property
    def are_models_loaded(self) -> bool:
        """
        Returns True if all inference models for the Deployment are loaded and ready
        to infer

        :return: True if all inference models for the deployed project are loaded in
            memory and ready for inference
        """
        return self._are_models_loaded

    def save(self, path_to_folder: Union[str, os.PathLike]):
        """
        Saves the Deployment instance to a folder on local disk

        :param path_to_folder: Folder to save the deployment to
        """
        project_dict = ProjectRESTConverter.to_dict(self.project)
        deployment_folder = os.path.join(path_to_folder, 'deployment')

        if not os.path.exists(deployment_folder):
            os.makedirs(deployment_folder)
        # Save project data
        project_filepath = os.path.join(deployment_folder, 'project.json')
        with open(project_filepath, 'w') as project_file:
            json.dump(project_dict, project_file, indent=4)
        # Save model for each task
        for task, model in zip(self.project.get_trainable_tasks(), self.models):
            model_dir = os.path.join(deployment_folder, task.title)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model.save(model_dir)

    @classmethod
    def from_folder(cls, path_to_folder: Union[str, os.PathLike]) -> 'Deployment':
        """
        Creates a Deployment instance from a specified `path_to_folder`

        :param path_to_folder: Path to the folder containing the Deployment data
        :return: Deployment instance corresponding to the deployment data in the folder
        """
        deployment_folder = path_to_folder
        if not path_to_folder.endswith("deployment"):
            if 'deployment' in os.listdir(path_to_folder):
                deployment_folder = os.path.join(path_to_folder, 'deployment')
            else:
                raise ValueError(
                    f"No `deployment` folder found in the directory at "
                    f"`{path_to_folder}`. Unable to load Deployment."
                )
        project_filepath = os.path.join(deployment_folder, 'project.json')
        with open(project_filepath, 'r') as project_file:
            project_dict = json.load(project_file)
        project = ProjectRESTConverter.from_dict(project_dict)
        task_folder_names = [task.title for task in project.get_trainable_tasks()]
        models: List[DeployedModel] = []
        for task_folder in task_folder_names:
            models.append(
                DeployedModel.from_folder(os.path.join(deployment_folder, task_folder))
            )
        return cls(models=models, project=project)

    def load_inference_models(self, device: str = 'CPU'):
        """
        Loads the inference models for the deployment to the specified device

        :param device: Device to load the inference models to
        """
        try:
            from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import (
                create_converter,
                IPredictionToAnnotationConverter
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
            model.load_inference_model(device=device)

            inference_converter = create_converter(
                converter_type=task.type.to_ote_domain(),
                labels=LabelSchemaEntity.from_labels(
                    labels=[
                        label.to_ote(task_type=task.type) for label in task.labels
                    ]
                )
            )
            inference_converters.update(
                {task.title: inference_converter}
            )
            empty_label = next((label for label in task.labels if label.is_empty), None)
            empty_labels.update({task.title: empty_label})

        self._inference_converters = inference_converters
        self._empty_labels = empty_labels
        self._are_models_loaded = True

    def infer(self, image: np.ndarray) -> Prediction:
        """
        Runs inference on an image for the full model chain in the deployment

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
                    image=image,
                    prediction=task_prediction,
                    rois=rois
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
        Runs pre-processing, inference, and post-processing on the input `image`, for
        the model associated with the `task`

        :param image: Image to run inference on
        :param task: Task to run inference for
        :return: Inference result
        """
        model = self._get_model_for_task(task)
        preprocessed_image, metadata = model.preprocess(image)
        inference_results = model.infer(preprocessed_image)
        postprocessing_results = model.postprocess(
            inference_results, metadata=metadata
        )
        converter = self._inference_converters[task.title]

        annotation_scene_entity = converter.convert_to_annotation(
            predictions=postprocessing_results,
            metadata=metadata
        )
        prediction = Prediction.from_ote(annotation_scene_entity)

        # Empty label is not generated by OTE correctly, append it here if there are
        # no other predictions
        if len(prediction.annotations) == 0:
            if self._empty_labels[task.title] is not None:
                prediction.append(
                    Annotation(
                        shape=Rectangle(x=0, y=0, width=1, height=1),
                        labels=[
                            ScoredLabel.from_label(
                                self._empty_labels[task.title], probability=1
                            )
                        ]
                    )
                )
        return prediction

    def _get_model_for_task(self, task: Task) -> DeployedModel:
        """
        Gets the DeployedModel instance corresponding to the input `task`

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
