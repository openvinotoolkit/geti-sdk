import json
import os
from typing import List, Union

import attr

import numpy as np

from sc_api_tools.data_models import Project, Task, TaskType, Prediction
from sc_api_tools.data_models.enums import OpenvinoModelName
from sc_api_tools.deployment.deployed_model import DeployedModel
from sc_api_tools.deployment.prediction_converters.detection import \
    convert_detection_output
from sc_api_tools.rest_converters import ProjectRESTConverter


@attr.s(auto_attribs=True)
class Deployment:
    """
    This class represents a deployed SC project that can be used to run inference
    locally
    """
    project: Project
    models: List[DeployedModel]

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
            json.dump(project_dict, project_file)
        # Save model for each task
        for task_index, model in enumerate(self.models):
            model_dir = os.path.join(deployment_folder, f'task_{task_index+1}')
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
        tasks = [
            folder for folder in os.listdir(deployment_folder)
            if folder.startswith('task')
        ]
        models: List[DeployedModel] = []
        for task_folder in tasks:
            models.append(
                DeployedModel.from_folder(os.path.join(deployment_folder, task_folder))
            )
        return cls(models=models, project=project)

    def load_inference_models(self, device: str = 'CPU'):
        """
        Loads the inference models for the deployment to the specified device

        :param device: Device to load the inference models to
        """
        for model, task in zip(self.models, self.project.get_trainable_tasks()):
            model_name = self._get_model_name(model, task)
            model.load_inference_model(model_name=model_name, device=device)

    def infer(
            self, image: np.ndarray
    ) -> List[Union[np.ndarray, Prediction]]:
        """
        Runs inference on an image for the full model chain in the deployment

        NOTE: For now this is only supported for detection and segmentation tasks

        :param image: Image to run inference on
        :return: inference results
        """
        result: List[Union[np.ndarray, Prediction]] = []
        if len(self.models) > 1:
            raise NotImplementedError(
                f"Running inference for a deployment of a task-chain project is not "
                f"yet supported. Please use inference on each individual model "
                f"instead."
            )
        for model, task in zip(self.models, self.project.get_trainable_tasks()):
            preprocessed_image, metadata = model.preprocess(image)
            inference_results = model.infer(preprocessed_image)
            postprocessing_results = model.postprocess(
                inference_results, metadata=metadata
            )
            if task.type == TaskType.DETECTION:
                result.append(
                    convert_detection_output(
                        model_output=postprocessing_results,
                        image_width=image.shape[1],
                        image_height=image.shape[0],
                        labels=task.labels
                    )
                )
            else:
                result.append(postprocessing_results)
        return result

    @staticmethod
    def _get_model_name(model: DeployedModel, task: Task) -> OpenvinoModelName:
        """
        Returns the name of the openvino model corresponding to the deployed `model`

        :param model: DeployedModel to get the name for
        :param task: Task corresponding to the model
        :return: OpenvinoModelName instance corresponding to the model name for `model`
        """
        task_type = task.type
        if task_type == TaskType.DETECTION:
            return OpenvinoModelName.SSD
        elif task_type == TaskType.CLASSIFICATION:
            raise NotImplementedError(
                "Running inference for classification models is not supported yet"
            )
            return OpenvinoModelName.OTE_CLASSIFICATION
        elif task_type == TaskType.ANOMALY_CLASSIFICATION:
            raise NotImplementedError(
                "Running inference for anomaly_classification models is not supported "
                "yet"
            )
            return OpenvinoModelName(model.name)
        elif task_type == TaskType.SEGMENTATION:
            # The proper name to return would be the 'class_name' variable, but
            # that doesn't work for `blur_segmentation` models yet. So we're always
            # returning segmentation for now
            model_name_parameter = model.hyper_parameters.get_parameter_by_name(
                "class_name"
            )
            class_name = OpenvinoModelName(model_name_parameter.value)
            return OpenvinoModelName.SEGMENTATION
