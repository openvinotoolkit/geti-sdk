# Copyright (C) 2023 Intel Corporation
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
from typing import List, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from geti_sdk import Geti
from geti_sdk.data_models import Prediction, Project
from geti_sdk.data_models.enums.task_type import TaskType
from geti_sdk.deployment import Deployment
from geti_sdk.rest_clients import ModelClient

from .utils import fit_pca_model, fre_score, perform_knn_indexing


class COODModel:
    """
    Out-of-distribution detection model. Uses the Combined out-of-distribution (COOD) detection
    algorithm (see : https://arxiv.org/abs/2403.06874).
    """

    def __init__(
        self,
        geti: Geti,
        project: Union[str, Project],
        deployment: Deployment = None,
    ):
        """
        todo[ood] : fill the docstring properly
        Combined Out-of-Distribution (COOD) detection model.
        :param geti: Geti instance on which the project to use for  lives
        :param project: Project or project name to use for the . The
            project must exist on the specified Geti instance
        :param deployment: Deployment to use for OOD dete. If not provided, the
        """
        self.geti = geti

        if isinstance(project, str):
            project_name = project
            self.project = geti.get_project(project_name=project_name)
        else:
            self.project = project

        self.model_client = ModelClient(
            session=geti.session, workspace_id=geti.workspace_id, project=self.project
        )

        # datasets_in_project = self.project.datasets
        #
        # self.image_client = ImageClient(
        #     session=geti.session, workspace_id=geti.workspace_id, project=project
        # )
        # path_to_save_data = "/Users/rgangire/workspace/Results/SDK/images_download"  # TODO[OOD]: Better directory.
        # self.image_client.download_all(
        #     path_to_folder=path_to_save_data, append_image_uid=True
        # )
        # # dataset_images = self.image_client.get_all_images()

        logging.info(
            f"Building Combined OOD detection model for Intel® Geti™ project `{self.project.name}`."
        )

        tasks_in_project = self.project.get_trainable_tasks()
        if len(tasks_in_project) != 1:
            raise ValueError(
                "Out-of-distribution detection models are only "
                "supported for projects with a single task for now."
            )
        # get the task type and check if it is classification
        task_type = tasks_in_project[0].task_type
        if task_type != TaskType.CLASSIFICATION:
            raise ValueError(
                "Out-of-distribution detection models are only "
                "supported for classification tasks for now."
            )

        if deployment is None:
            self.deployment = self._get_usable_deployment()
        else:
            if not deployment.models[0].has_xai_head:
                raise ValueError(
                    "The provided deployment does not have an model with an XAI head."
                    "Please reconfigure the deployment to include a model with an XAI head "
                    "(OptimizedModel.has_xai_head must be True)"
                )

            self.deployment = deployment

        if not self.deployment.are_models_loaded:
            self.deployment.load_inference_models(device="CPU")

        # The COOD random forest classifier
        self.ood_classifier = None

        # A dict consisting smaller OOD models (FRE, EnWeDi, etc)
        self.sub_models = {
            "knn_based": KNNBasedOODModel(knn_k=10),
            "class_fre": ClassFREModel(n_components=0.995),
            "max_softmax_probability": MaxSoftmaxProbabilityModel(),
        }

    def _get_usable_deployment(self) -> Deployment:
        """
        Get a deployment that has an optimised model with an XAI head.
        """
        # Check if there's at least one trained model in the project
        models = self.model_client.get_all_active_models()
        if len(models) == 0:
            raise ValueError(
                "No trained models were found in the project, please either "
                "train a model first or specify an algorithm to train."
            )

        # We need the model which has xai enabled - this allows us to get the feature vector from the model.
        model_with_xai_head = None

        for model in models:
            for optimised_model in model.optimized_models:
                if optimised_model.has_xai_head:
                    model_with_xai_head = optimised_model
                    break

        if model_with_xai_head is None:
            raise ValueError(
                "No trained model with an XAI head was found in the project, "
                "please train a model with an XAI head first."
            )

        deployment = self.geti.deploy_project(
            project_name=self.project.name, models=[model_with_xai_head]
        )
        return deployment

    def train(self):
        """
        Train the COOD model using the RandomForestClassifier
        """
        # Step 1 : ID Images
        #       1a : Get labelled images from the project
        # Step 2 : OOD Data
        #       2a : Check if any dataset called as OOD images exist
        #       2b : Else, generate images by applying corruptions
        # Step 3 : Extract Features, and predictions
        #       3a : Find a xai model
        # Step 4 :Train all the sub-models (FRE,EnWeDi, etc)
        # Step 5 : Forward pass through sub-models to get ood scores for each image
        # Step 6 : Train the COOD Random Forest
        # Step 7 : Test COOD on test set (?)  Determine threshold (usually this is just 0.5)

        # Step 4 : Train all the sub models

        for sub_model in self.sub_models.values():
            sub_model.train()

        ood_classifier = RandomForestClassifier()
        features = []  # Each element is an output (ood score) from the sub-models
        labels = []  # OOD = 1, ID = 0
        ood_classifier.fit(features, labels)

        self.ood_classifier = ood_classifier

    def __call__(self, prediction: Prediction) -> float:
        """
        Return the COOD Score based using feature vector and prediction probabilities in "prediction".
        """
        # feature_vector = prediction.feature_vector
        # for annotation in prediction.annotations:
        #     # Find a Better way to get probability (or logits_)
        #     prediction_probabilities = [
        #         label.probability for label in annotation.labels
        #     ]
        #

        cood_features = self.call_sub_models(prediction)
        cood_score = self.ood_classifier.predict(cood_features)
        return cood_score

    def call_sub_models(self, prediction: Prediction) -> np.ndarray:
        """
        Call the sub-models to get the OOD scores
        """
        # see paper at https://github.com/VitjanZ/DRAEM
        # Call's all submodel objects. Gets back individual scores
        pass

    def _get_labeled_id_images_from_project(self):
        """
        Create a list of the images that will be ID
        """
        pass

    def _get_ood_images_from_project(self):
        """
        Create a list of the images that will be OOD
        """
        pass

    def _create_ood_images(self):
        """
        Create near-OOD images by applying strong corruptions to the in-distribution images
        """
        # Options  : Applying corruptions, generating Perlin Noise Images, Background extraction
        pass


class KNNBasedOODModel:
    """
    k Nearest Neighbour based OOD detection model.
    """

    def __init__(self, knn_k: int = 10):
        self.knn_k = knn_k
        self.knn_search_index = None

    def train(self):
        """
        Train the kNN model
        """
        feature_vectors = None
        self.knn_search_index = perform_knn_indexing(feature_vectors)

    def __call__(self, prediction: Prediction) -> dict:
        """
        Return the kNN OOD score for the given feature vector.
        """
        pass


class ClassFREModel:
    """
    Yet to be finalised
    """

    def __init__(self, n_components=0.995):
        self.n_components = n_components
        self.pca_models_per_class = {}

    def train(self):
        """
        Fit PCA Models on the in-distribution data for each class.
        """
        # iterate through unique labels and fit pca model for each class
        pca_models = {}
        features: np.ndarray = None
        labels: List[str] = None

        for label in np.unique(labels):
            # labels are list of class names and not indices
            class_indices = [i for i, j in enumerate(labels) if j == label]
            class_features = features[class_indices]
            pca_models[label] = fit_pca_model(
                feature_vectors=class_features, n_components=self.n_components
            )

        self.pca_models_per_class = pca_models

    def __call__(self, prediction: Prediction) -> dict:
        """
        Return the class fre score for the given feature vector.
        """
        features = prediction.feature_vector
        fre_scores_per_class = {}
        # class_fre_models is a dict with label name and pca model.
        for label, pca_model in self.class_fre_models.items():
            fre_scores_per_class[label] = fre_score(
                feature_vectors=features, pca_model=pca_model
            )

        # return maximum FRE
        return {"max_fre": max(fre_scores_per_class.values())}


class MaxSoftmaxProbabilityModel:
    """
    Maximum Softmax Probability Model - A baseline OOD detection model.
    Use the concept that a lower maximum softmax probability indicates that the image could be OOD.
    See
    """

    def __init__(self):
        pass


class DKNNModel:
    """
    todo[ood] : Docstring if this class is actually used
    """

    # This will be called by KNNBasedModel.
    # KnnBasedModel would have prepared the index. COOD or OODSubModel would have prepared the feature vectors
    pass
