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
import logging
import os
import shutil
import tempfile
import time
import zipfile
from typing import Dict, List, Optional, Sequence, Union

from geti_sdk.data_models import Project
from geti_sdk.data_models.code_deployment_info import (
    CodeDeploymentInformation,
    DeploymentModelIdentifier,
)
from geti_sdk.data_models.enums import DeploymentState, OptimizationType
from geti_sdk.data_models.model import Model, OptimizedModel
from geti_sdk.deployment import Deployment
from geti_sdk.http_session import GetiSession
from geti_sdk.utils import deserialize_dictionary, get_supported_algorithms

from .configuration_client import ConfigurationClient
from .model_client import ModelClient
from .prediction_client import PredictionClient


class DeploymentClient:
    """
    Class to manage model deployment for a certain Intel® Geti™ project.
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/projects/{project.id}"
        self.supported_algos = get_supported_algorithms(session)
        self._deployment_resource_cache: Dict[str, str] = {}

        self._model_client = ModelClient(
            workspace_id=workspace_id, project=project, session=session
        )
        self._prediction_client = PredictionClient(
            workspace_id=workspace_id, project=project, session=session
        )
        self._configuration_client = ConfigurationClient(
            workspace_id=workspace_id, project=project, session=session
        )

    @property
    def code_deployment_url(self) -> str:
        """
        Return the base URL for the code deployment group of endpoints

        :return: URL for the code deployment endpoints for the Intel® Geti™ project
        """
        return self.base_url + "/code_deployments"

    @property
    def ready_to_deploy(self) -> bool:
        """
        Return True when the project is ready for deployment, False otherwise.

        A project is ready for deployment when it contains at least one trained model
        for each task.

        :return: True when the project is ready for deployment, False otherwise
        """
        return self._prediction_client.ready_to_predict

    def _request_deployment(
        self, model_identifiers: Sequence[DeploymentModelIdentifier]
    ) -> CodeDeploymentInformation:
        """
        Request the creation of a code deployment for the project, using the models
        with identifiers as passed in `models`

        :param model_identifiers: List of identifiers for the models to use in the
            deployment
        :return: CodeDeploymentInformation object, holding all available information
            regarding the state of the deployment creation process on the Intel® Geti™
            server.
        """
        model_id_data = [model.to_dict() for model in model_identifiers]
        response = self.session.get_rest_response(
            url=self.code_deployment_url + ":prepare",
            method="POST",
            data={"models": model_id_data},
        )
        logging.info(f"Deployment for project '{self.project.name}' started.")
        return deserialize_dictionary(response, CodeDeploymentInformation)

    def _get_deployment_status(self, deployment_id: str) -> CodeDeploymentInformation:
        """
        Return the actual status of a code deployment creation process on the Intel®
        Geti™ server.

        :param deployment_id: Unique identifier of the deployment to get the status for
        :return: CodeDeploymentInformation instance holding the current state of the
            deployment creation process
        """
        response = self.session.get_rest_response(
            url=self.code_deployment_url + f"/{deployment_id}", method="GET"
        )
        return deserialize_dictionary(response, CodeDeploymentInformation)

    def _fetch_deployment(self, deployment_id: str) -> Deployment:
        """
        Download and return the deployment identified by `deployment_id`. If the
        deployment is not ready to be downloaded yet, this method will raise an error.

        :param deployment_id: Unique identifier of the deployment to download
        :return: `Deployment` object containing the deployment
        """
        status = self._get_deployment_status(deployment_id=deployment_id)
        if status.state != DeploymentState.DONE:
            raise ValueError(
                f"Deployment with ID '{deployment_id}' is not ready to be retrieved "
                f"(yet). The current state is '{status.state}'. Deployment can only be "
                f"retrieved when it is in state 'DONE'. "
            )
        deployment_tempdir = tempfile.mkdtemp()
        self._deployment_resource_cache.update({deployment_id: deployment_tempdir})
        zipfile_path = os.path.join(deployment_tempdir, "deployment.zip")
        response = self.session.get_rest_response(
            url=self.code_deployment_url + f"/{deployment_id}/download",
            method="GET",
            contenttype="zip",
        )
        logging.info("Downloading project deployment archive...")
        with open(zipfile_path, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(zipfile_path, "r") as zipped_deployment:
            zipped_deployment.extractall(deployment_tempdir)
        logging.info(
            f"Deployment for project '{self.project.name}' downloaded and extracted "
            f"successfully."
        )
        if os.path.exists(zipfile_path):
            os.remove(zipfile_path)
        return Deployment.from_folder(
            path_to_folder=os.path.join(deployment_tempdir, "deployment")
        )

    def deploy_project(
        self,
        output_folder: Optional[Union[str, os.PathLike]] = None,
        models: Optional[Sequence[Union[Model, OptimizedModel]]] = None,
    ) -> Deployment:
        """
        Deploy a project by creating a Deployment instance. The Deployment contains
        the optimized active models for each task in the project, and can be loaded
        with OpenVINO to run inference locally.

        The `models` parameter can be used in two ways:

          - If `models` is left as None this method will create a deployment containing
            the current active model for each task in the project.

          - If a list of models is passed, then it should always contain a number of
            Models less than or equal to the amount of tasks in the project. For
            instance, for a task-chain project with two tasks a list of at most two
            models could be passed (one for each task).
            If in this case a list of only one model is passed, that model will be
            used for its corresponding task and the method will resort to using the
            active model for the other task in the project.

        :param output_folder: Path to a folder on local disk to which the Deployment
            should be downloaded. If no path is specified, the deployment will not be
            saved to disk directly. Note that it is always possible to save the
            deployment once it has been created, using the `deployment.save` method.
        :param models: Optional list of models to use in the deployment. If no list is
            passed, this method will create a deployment using the currently active
            model for each task in the project.
        :return: Deployment for the project
        """
        if not self.ready_to_deploy:
            raise ValueError(
                f"Project '{self.project.name}' is not ready for deployment, please "
                f"make sure that each task in the project has at least one model "
                f"trained before deploying the project."
            )

        # Get the models to deploy
        tasks = self.project.get_trainable_tasks()
        if models is None:
            models = self._model_client.get_all_active_models()

        tasks_with_model = [
            self._model_client.get_task_for_model(model) for model in models
        ]

        if len(models) < len(tasks):

            tasks_without_model = [
                task for task in tasks if task not in tasks_with_model
            ]
            active_models = [
                self._model_client.get_active_model_for_task(task)
                for task in tasks_without_model
            ]

            models = models + active_models

            model_id_to_task_id = {
                model.id: task.id
                for model, task in zip(models, tasks_with_model + tasks_without_model)
            }
        else:
            model_id_to_task_id = {
                model.id: task.id for model, task in zip(models, tasks_with_model)
            }

        # Fetch the optimized model for each model, if it is not optimized already
        optimized_models: List[OptimizedModel] = []
        for model in models:
            if not isinstance(model, OptimizedModel):
                current_optim_models = model.optimized_models
                optimization_types = [
                    op_model.optimization_type for op_model in current_optim_models
                ]
                preferred_model = current_optim_models[0]
                for optimization_type in OptimizationType:
                    if optimization_type in optimization_types:
                        preferred_model = current_optim_models[
                            optimization_types.index(optimization_type)
                        ]
                        break
                optimized_models.append(preferred_model)

                task_id_for_model = model_id_to_task_id[model.id]
                model_id_to_task_id.update({preferred_model.id: task_id_for_model})
            else:
                optimized_models.append(model)

        model_identifiers = [
            DeploymentModelIdentifier.from_model(model) for model in optimized_models
        ]
        # Make sure models are sorted according to task order
        sorted_optimized_models: List[OptimizedModel] = []
        for task in tasks:
            for model in optimized_models:
                model_task_id = model_id_to_task_id[model.id]
                if model_task_id == task.id:
                    sorted_optimized_models.append(model)
                    break

        # Retrieve hyper parameters for the models
        hyper_parameters = [
            self._configuration_client.get_for_model(
                model_id=model.id, task_id=model_id_to_task_id[model.id]
            )
            for model in sorted_optimized_models
        ]

        # Make the request to prepare the deployment
        code_deployment = self._request_deployment(model_identifiers=model_identifiers)

        # Wait for the deployment to become available
        stop_polling = False
        logging.info("Waiting for the deployment to be created...")
        while not stop_polling:
            code_deployment = self._get_deployment_status(code_deployment.id)
            if code_deployment.state == DeploymentState.DONE:
                stop_polling = True
            elif code_deployment.state == DeploymentState.FAILED:
                raise ValueError(
                    f"The Intel® Geti™ server failed to create deployment for "
                    f"project '{self.project.name}'."
                )
            time.sleep(1)

        # Fetch the deployment package
        deployment = self._fetch_deployment(deployment_id=code_deployment.id)

        # Attach the appropriate hyper parameters to the deployed models
        for index, model in enumerate(deployment.models):
            model.hyper_parameters = hyper_parameters[index]

        # Save the deployment, if needed
        if output_folder is not None:
            deployment.save(path_to_folder=output_folder)

            # Reload the deployment, and remove any temporary files left behind
            deployment_from_disk = Deployment.from_folder(
                path_to_folder=os.path.join(output_folder, "deployment")
            )
            self._clean_up_temporary_resources(deployment_id=code_deployment.id)
            return deployment_from_disk
        return deployment

    def _clean_up_temporary_resources(self, deployment_id: str) -> bool:
        """
        Clean up any temporary files that were used to create the deployment with id
        `deployment_id`

        Note that resources can only be removed by the same DeploymentClient instance
        that created the deployment

        :param deployment_id: ID of the deployment for which temporary resources
            should be removed
        :return True if all temporary resources were removed successfully, False
            otherwise
        """
        if deployment_id not in self._deployment_resource_cache.keys():
            return False
        temp_dir = self._deployment_resource_cache[deployment_id]
        if os.path.isdir(temp_dir):
            shutil.rmtree(self._deployment_resource_cache[deployment_id])
            return True
        return False

    def __del__(self):
        """
        Clean up the temporary directories created to store deployments. This
        method is called when the DeploymentClient instance is deleted.
        """
        for deployment_id, temp_dir in self._deployment_resource_cache.items():
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)
