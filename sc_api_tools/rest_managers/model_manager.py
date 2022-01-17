import json
import os
from typing import List, Optional, Union, TypeVar

from sc_api_tools.data_models import (
    Project,
    ModelGroup,
    Model,
    OptimizedModel,
    ModelSummary, Task
)
from sc_api_tools.data_models.enums import Domain
from sc_api_tools.http_session import SCSession
from sc_api_tools.rest_converters import ModelRESTConverter
from sc_api_tools.utils import get_supported_algorithms

ModelType = TypeVar("ModelType", Model, OptimizedModel)


class ModelManager:
    """
    Class to manage the models and model groups for a certain project
    """

    def __init__(self, workspace_id: str, project: Project, session: SCSession):
        self.session = session
        project_id = project.id
        self.project = project
        self.workspace_id = workspace_id
        self.task_ids = [task.id for task in project.get_trainable_tasks()]
        self.base_url = f"workspaces/{workspace_id}/projects/{project_id}/" \
                        f"model_groups"

    def get_all_model_groups(self) -> List[ModelGroup]:
        """
        Returns a list of all model groups in the project

        :return: List of model groups in the project
        """
        response = self.session.get_rest_response(
            url=self.base_url,
            method="GET"
        )
        model_groups = [
            ModelRESTConverter.model_group_from_dict(group) for group in response
        ]
        # Update algorithm details
        supported_algos = get_supported_algorithms(self.session)
        for group in model_groups:
            group.algorithm = supported_algos.get_by_model_template(
                model_template_id=group.model_template_id
            )
        return model_groups

    def get_model_group_by_algo_name(self, algorithm_name: str) -> Optional[ModelGroup]:
        """
        Returns the model group for the algorithm named `algorithm_name`, if any. If
        no model group for this algorithm is found in the project, this method returns
        None

        :param algorithm_name: Name of the algorithm
        :return: ModelGroup instance corresponding to this algorithm
        """
        model_groups = self.get_all_model_groups()
        return next(
            (
                group for group in model_groups
                if group.algorithm.algorithm_name == algorithm_name
            ), None
        )

    def _get_model_detail(self, group_id: str, model_id: str) -> Model:
        """
        Returns the Model object holding detailed information about the model with
        `model_id` living in the model group with `group_id`

        :param group_id: Unique database ID of the model group to which the model to
            get belongs
        :param model_id: Unique database ID of the model to get
        :return: Model instance holding detailed information about the model
        """
        model_detail = self.session.get_rest_response(
            url=f"{self.base_url}/{group_id}/models/{model_id}",
            method="GET"
        )
        model = ModelRESTConverter.model_from_dict(model_detail)
        model.model_group_id = group_id
        return model

    def get_active_model_for_task(self, task: Task) -> Optional[Model]:
        """
        Returns the Model details for the currently active model, for a task if any.
        If the task does not have any trained models, this method returns None

        :param task: Task object containing details of the task to get the model for
        :return: Model object representing the currently active model in the SC
            project, if any
        """
        model_groups = self.get_all_model_groups()
        model_id: Optional[str] = None
        group_id: Optional[str] = None
        for group in model_groups:
            if not group.has_trained_models:
                continue
            if group.algorithm.domain != Domain.from_task_type(task.type):
                continue
            model_summary = group.get_latest_model()
            if model_summary is not None:
                if model_summary.active_model:
                    model_id = model_summary.id
                    group_id = group.id
                    break
        if model_id is not None:
            return self._get_model_detail(group_id=group_id, model_id=model_id)
        return None

    def _download_model(
            self, model: ModelType, path_to_folder: str
    ) -> ModelType:
        """
        Downloads a Model or OptimizedModel

        :param model: Model or OptimizedModel to download
        :return: Model or OptimizedModel object holding the details of the downloaded
            model
        """
        if isinstance(model, Model):
            url = f"{self.base_url}/{model.model_group_id}/models/{model.id}/export"
            filename = f"{model.name}_base.zip"
        else:
            url = f"{self.base_url}/{model.model_group_id}/models/" \
                  f"{model.previous_trained_revision_id}/optimized_models/" \
                  f"{model.id}/export"
            filename = f"{model.name}_{model.optimization_type}_optimized.zip"
        response = self.session.get_rest_response(
            url=url,
            method="GET",
            contenttype="zip"
        )
        model_folder = os.path.join(path_to_folder, 'models')
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_filepath = os.path.join(model_folder, filename)
        with open(model_filepath, 'wb') as f:
            f.write(response.content)
        return model

    def download_active_model_for_task(
            self, path_to_folder: str, task: Task
    ) -> Optional[Model]:
        """
        Downloads the currently active model for the task
        If the task does not have an active model yet, this method returns None

        This method will create a directory 'models' in the path specified in
        `path_to_folder`

        :param path_to_folder: Path to the target folder in which to save the active
            model, and all optimized models derived from it.
        :param task: Task object containing details of the task to download the model
            for
        :return: Model instance holding the details of the active model
        """
        model = self.get_active_model_for_task(task=task)
        if model is None:
            print(
                f"Project '{self.project.name} does not have any trained models yet, "
                f"unable to download active model."
            )
            return None
        model_filepath = os.path.join(path_to_folder, 'models')
        print(
            f"Downloading active model for task {task.title} in project "
            f"{self.project.name} to folder {model_filepath}..."
        )
        self._download_model(model, path_to_folder=path_to_folder)
        for optimized_model in model.optimized_models:
            self._download_model(optimized_model, path_to_folder=path_to_folder)
        model_info_filepath = os.path.join(
            model_filepath, f"{task.type}_model_details.json"
        )
        with open(model_info_filepath, 'w') as f:
            json.dump(model.to_dict(), f, indent=4)
        return model

    def get_all_active_models(self) -> List[Optional[Model]]:
        """
        Returns the Model details for the active model for all tasks in the project,
        if the tasks have any.

        This method returns a list of Models, where the index of the Model in the list
        corresponds to the index of the task in list of trainable tasks for the project.

        If any of the tasks do not have a trained model, the entry corresponding to
        the index of that task will be None

        :return: Model object representing the currently active model for the task in
            the SC project, if any
        """
        return [
            self.get_active_model_for_task(task=task)
            for task
            in self.project.get_trainable_tasks()
        ]

    def download_all_active_models(self, path_to_folder: str) -> List[Optional[Model]]:
        """
        Downloads the active models for all tasks in the project.

        This method will create a directory 'models' in the path specified in
        `path_to_folder`

        :param path_to_folder: Path to the target folder in which to save the active
            models, and all optimized models derived from them.
        :return: List of Model objects representing the currently active models
            (if any) for all tasks in the SC project. The index of the Model in the
            list corresponds to the index of the task in the list of trainable tasks
            for the project.
        """
        return [
            self.download_active_model_for_task(
                path_to_folder=path_to_folder, task=task
            )
            for task
            in self.project.get_trainable_tasks()
        ]
