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

import json
import logging
import os

# import time
from typing import List, Union

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from geti_sdk import Geti
from geti_sdk.data_models import Prediction, Project
from geti_sdk.data_models.enums.task_type import TaskType
from geti_sdk.data_models.project import Dataset
from geti_sdk.deployment import Deployment
from geti_sdk.rest_clients import AnnotationClient, ImageClient, ModelClient

from .utils import (  # normalise_features,
    CutoutTransform,
    fit_pca_model,
    fre_score,
    perform_knn_indexing,
    perform_knn_search,
)

ID_DATASET_NAMES = ["Dataset"]
OOD_DATASET_NAMES = ["ood dataset"]


class DistributionDataItem:
    """
    A class to store the data for the COOD model.
    """

    def __init__(
        self,
        media_name: str,
        image_path: str,
        annotated_label: str | None,
        feature_vector: np.ndarray,
        prediction_probability: float,
        predicted_label: str,
        raw_prediction: Prediction,
    ):
        self.media_name = media_name
        self.image_path = image_path
        self.annotated_label = annotated_label
        self.feature_vector = feature_vector
        self.prediction_probability = prediction_probability
        self.predicted_label = predicted_label
        self.raw_prediction = raw_prediction

    # TODO[OOD] : Take only required fields and everything else can be property def where they can be
    #  extracted from raw_prediction
    # TODO[OOD] : Normalise feature vector in efficient ways when doing the above todo task


class COODModel:
    """
    Out-of-distribution detection model. Uses the Combined out-of-distribution (COOD) detection
    algorithm (see : https://arxiv.org/abs/2403.06874).

    Workspace directory for this Model follows the structure:
    ```
    workspace_dir
    ├── data
    │   ├── ID
    │   │   ├── images
    │   │   ├── annotations
    │   │   └── predictions
    │   └── OOD
    │       ├── images
    │       └── predictions
    └── models
        ├── deployment_model
        └── ood_model

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
        # TODO[OOD] : move any possible methods to Utils
        self.geti = geti
        self.distribution_data = {}
        # TODO[ood] Once the id_data is made into a class object,
        # this can be a list, with each data having a distribution name

        # TODO[OOD] : Features are not yet normalised

        # TODO[OOD] : Make it tmpdir or something, make it a workspace dire with model, data subdirs
        self.workspace_dir = "/Users/rgangire/workspace/Results/SDK/"
        self.data_dir = os.path.join(self.workspace_dir, "data")

        if isinstance(project, str):
            project_name = project
            self.project = geti.get_project(project_name=project_name)
        else:
            self.project = project

        self._prepare_geti_clients()

        self.corruption = CutoutTransform()

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

        self._prepare_id_odd_data()

        # The COOD random forest classifier
        self.ood_classifier = None

        # A dict consisting smaller OOD models (FRE, EnWeDi, etc)
        self.sub_models = {
            "knn_based": KNNBasedOODModel(knn_k=10),
            "class_fre": ClassFREModel(n_components=0.995),
            "max_softmax_probability": MaxSoftmaxProbabilityModel(),
        }

        self._train_sub_models()
        self.train()

    def _prepare_geti_clients(self):
        """
        Prepare the Geti clients for the project
        """
        self.model_client = ModelClient(
            session=self.geti.session,
            workspace_id=self.geti.workspace_id,
            project=self.project,
        )
        self.image_client = ImageClient(
            session=self.geti.session,
            workspace_id=self.geti.workspace_id,
            project=self.project,
        )

        self.annotation_client = AnnotationClient(
            session=self.geti.session,
            workspace_id=self.geti.workspace_id,
            project=self.project,
        )

    def _prepare_id_odd_data(self):
        datasets_in_project = self.project.datasets

        id_datasets = []
        ood_datasets = []

        for dataset in datasets_in_project:
            if dataset.name in ID_DATASET_NAMES:
                id_datasets.append(dataset)
            elif dataset.name in OOD_DATASET_NAMES:
                ood_datasets.append(dataset)

        if len(id_datasets) == 0:
            raise ValueError(
                "Could not find any relevant datasets for in-distribution data. "
                "Please make sure that the project contains at least one dataset with the names: "
                f"{ID_DATASET_NAMES}."
            )

        id_data = []
        for dataset in id_datasets:
            id_data.extend(self._prepare_data_from_dataset(dataset))

        ood_data = []
        if len(ood_datasets) == 0:
            logging.info(
                "No out-of-distribution datasets found in the project. "
                "Generating near-OOD images by applying strong corruptions to the in-distribution images."
            )
            for dataset in id_datasets:
                ood_path = self._create_ood_images(dataset)
                ood_data.extend(self._prepare_ood_data(ood_dataset_path=ood_path))

        else:
            for dataset in ood_datasets:
                ood_data.extend(self._prepare_data_from_dataset(dataset))

        # Len of id_data and ood_data
        logging.info(f"Number of in-distribution samples: {len(id_data)}")
        logging.info(f"Number of out-of-distribution samples: {len(ood_data)}")

        self.distribution_data = {
            "id_data": id_data,
            "ood_data": ood_data,
        }

    def _train_sub_models(self):
        """
        Initialise the OOD models
        """
        for sub_model in self.sub_models.values():
            sub_model.train(distribution_data=self.distribution_data)

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

        # TODO[OOD] : Take the model which has highest accuracy or some other metric, instead of the first model
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

        ood_output_sub_models = {}
        for sub_model in self.sub_models.values():
            if sub_model.is_trained:
                ood_output_sub_models[sub_model.__class__.__name__] = sub_model(
                    self.distribution_data["ood_data"]
                )

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

    def _prepare_data_from_dataset(
        self, dataset: Dataset
    ) -> List[DistributionDataItem]:
        required_data_all = []
        dataset_name = dataset.name
        dataset_dir = os.path.join(self.data_dir, dataset_name)

        media_list = self.image_client.get_all_images(dataset=dataset)
        self.image_client.download_all(
            path_to_folder=dataset_dir, append_image_uid=True
        )
        self.annotation_client.download_annotations_for_images(
            images=media_list,
            path_to_folder=dataset_dir,
            append_image_uid=True,
        )

        # For each image in the dataset, annotation (if exists) is read and feature vector is extracted
        annotations_dir = os.path.join(dataset_dir, "annotations")
        image_dir = os.path.join(dataset_dir, "images")
        for media in media_list:
            required_data = {}
            media_filename = media.name + "_" + media.id
            annotation_file = os.path.join(annotations_dir, f"{media_filename}.json")
            image_path = os.path.join(image_dir, f"{media_filename}.jpg")

            required_data["media_name"] = media_filename
            required_data["image_path"] = image_path

            # TODO[OOD]": Careful with this. This has to be made better. Check for only annotation_kind = "ANNOTATION"
            # only then add it as "annotation_label"
            # make the list of dictionary a object - use the iLRF dataset item class
            if os.path.exists(annotation_file):
                with open(annotation_file, "r") as f:
                    annotation = json.load(f)
                    required_data["annotated_label"] = annotation["annotations"][0][
                        "labels"
                    ][0]["name"]
            else:
                required_data["annotated_label"] = None

            prediction = self._infer(image_path=image_path, explain=True)
            feature_vector = prediction.feature_vector
            if len(feature_vector.shape) != 1:
                feature_vector = feature_vector.flatten()
            required_data["feature_vector"] = feature_vector
            required_data["prediction_probability"] = (
                prediction.annotations[0].labels[0].probability
            )
            required_data["predicted_label"] = prediction.annotations[0].labels[0].name

            data_item = DistributionDataItem(
                media_name=required_data["media_name"],
                image_path=required_data["image_path"],
                annotated_label=required_data["annotated_label"],
                feature_vector=required_data["feature_vector"],
                prediction_probability=required_data["prediction_probability"],
                predicted_label=required_data["predicted_label"],
                raw_prediction=prediction,
            )

            required_data_all.append(data_item)

        return required_data_all

    def _infer(self, image_path: str, explain: bool = False) -> Prediction:
        """
        Infer the image and get the prediction using the deployment
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if explain:
            # Note that a check to see if xai model is present in the deployment is not done.
            # If the model is not present, then feature_vector will be None
            return self.deployment.explain(image=img_rgb)
        else:
            return self.deployment.infer(image=img_rgb)

    def _create_ood_images(self, reference_dataset: Dataset) -> str:
        """
        Create near-OOD images by applying strong corruptions to the in-distribution images in the reference datasets.
        """
        # Options  : Applying corruptions, generating Perlin Noise Images, Background extraction (using saliency maps)
        ref_images_path = os.path.join(self.data_dir, reference_dataset.name, "images")
        corrupted_images_path = os.path.join(self.data_dir, "ood_images")
        if not os.path.exists(corrupted_images_path):
            os.makedirs(corrupted_images_path)

        for image_name in os.listdir(ref_images_path):
            image_path = os.path.join(ref_images_path, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            corrupted_img = self.corruption(img)
            corrupted_image_path = os.path.join(corrupted_images_path, image_name)
            cv2.imwrite(corrupted_image_path, corrupted_img)

        return corrupted_images_path

    def _prepare_ood_data(self, ood_dataset_path):
        """
        Prepare the OOD data for the model
        """
        ood_data_all = []
        for file_name in os.listdir(ood_dataset_path):
            required_data = {}

            required_data["image_path"] = os.path.join(ood_dataset_path, file_name)
            required_data["media_name"] = os.path.splitext(os.path.basename(file_name))[
                0
            ]
            required_data["annotated_label"] = None
            prediction = self._infer(
                image_path=os.path.join(ood_dataset_path, file_name), explain=True
            )
            feature_vector = prediction.feature_vector
            if len(feature_vector.shape) != 1:
                feature_vector = feature_vector.flatten()
            required_data["feature_vector"] = feature_vector
            required_data["prediction_probability"] = (
                prediction.annotations[0].labels[0].probability
            )
            required_data["predicted_label"] = prediction.annotations[0].labels[0].name

            data_item = DistributionDataItem(
                media_name=required_data["media_name"],
                image_path=required_data["image_path"],
                annotated_label=required_data["annotated_label"],
                feature_vector=required_data["feature_vector"],
                prediction_probability=required_data["prediction_probability"],
                predicted_label=required_data["predicted_label"],
                raw_prediction=prediction,
            )

            ood_data_all.append(data_item)
        return ood_data_all


class KNNBasedOODModel:
    """
    k Nearest Neighbour based OOD detection model.
    """

    def __init__(self, knn_k: int = 10):
        self.knn_k = knn_k
        self.knn_search_index = None
        self._is_trained = False

    def train(self, distribution_data: dict):
        """
        Train the kNN model
        """
        id_data = distribution_data["id_data"]
        feature_vectors = np.array([data["feature_vector"] for data in id_data])
        self.knn_search_index = perform_knn_indexing(feature_vectors, use_gpu=False)
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        """
        Return True if the model is trained.
        """
        return self._is_trained

    def __call__(self, prediction: Prediction) -> dict:
        """
        Return the kNN OOD score for the given feature vector.
        """
        features = prediction.feature_vector
        ood_scores = perform_knn_search(
            knn_search_index=self.knn_search_index,
            feature_vectors=features,
            k=self.knn_k,
        )
        # distance to the kth nearest neighbour
        return {"knn_ood_score": ood_scores[:, -1]}


class ClassFREModel:
    """
    Yet to be finalised
    """

    def __init__(self, name: str = "class_fre_model", n_components=0.995):
        self.n_components = n_components
        self.pca_models_per_class = {}
        self._is_trained = False

    def train(self, distribution_data: dict):
        """
        Fit PCA Models on the in-distribution data for each class.
        """
        id_data = distribution_data["id_data"]
        features = np.array([data["feature_vector"] for data in id_data])
        labels = np.array([data["annotated_label"] for data in id_data])

        # iterate through unique labels and fit pca model for each class
        pca_models = {}

        for label in np.unique(labels):
            # labels are list of class names and not indices
            class_indices = [i for i, j in enumerate(labels) if j == label]
            class_features = features[class_indices]
            pca_models[label] = fit_pca_model(
                feature_vectors=class_features, n_components=self.n_components
            )

        self.pca_models_per_class = pca_models
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        """
        Return True if the model is trained.
        """
        return self._is_trained

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
        self._is_trained = False

    def train(self, distribution_data: dict):
        """
        MSP model does not require training.
        """
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        """
        Return True if the model is trained.
        """
        return self._is_trained

    def __call__(self, prediction: Prediction) -> float:
        """
        Return the maximum softmax probability for the given prediction.
        """
        prediction_probabilities = [
            label.probability for label in prediction.annotations[0].labels
        ]
        return max(prediction_probabilities)


class DKNNModel:
    """
    todo[ood] : Docstring if this class is actually used
    """

    # This will be called by KNNBasedModel.
    # KnnBasedModel would have prepared the index. COOD or OODSubModel would have prepared the feature vectors
    pass
