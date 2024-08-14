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
import tempfile
from abc import ABCMeta, abstractmethod
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

from .utils import (
    CutoutTransform,
    calculate_entropy_nearest_neighbours,
    fit_pca_model,
    fre_score,
    get_usable_deployment,
    normalise_features,
    perform_knn_indexing,
    perform_knn_search,
)

ID_DATASET_NAMES = ["Dataset"]
OOD_DATASET_NAMES = ["ood dataset"]


# TODO[ood] Once the id_data is made into a class object,


class DistributionDataItem:
    """
    A class to store the data for the COOD model.
    """

    def __init__(
        self,
        raw_prediction: Prediction,
        media_name: Union[str, None],
        media_path: Union[str, None],
        annotated_label: Union[str, None],
        normalise_feature_vector: bool = True,
    ):
        self.media_name = media_name
        self.image_path = media_path
        self.annotated_label = annotated_label
        self.raw_prediction = raw_prediction
        self._normalise_feature_vector = normalise_feature_vector

        feature_vector = raw_prediction.feature_vector

        if len(feature_vector.shape) != 1:
            feature_vector = feature_vector.flatten()

        if normalise_feature_vector:
            feature_vector = normalise_features(feature_vector)[0]

        self.feature_vector = feature_vector
        self.max_prediction_probability = (
            raw_prediction.annotations[0].labels[0].probability,
        )
        self.predicted_label = (raw_prediction.annotations[0].labels[0].name,)

    @property
    def is_feature_vector_normalised(self) -> bool:
        """
        Return True if the feature vector is normalised.
        """
        return self._normalise_feature_vector


class COODModel:
    """
    Out-of-distribution detection model. Uses the Combined out-of-distribution (COOD) detection
    algorithm (see : https://arxiv.org/abs/2403.06874).

    Uses a temporary directory for storing data with the following structure:
    ```
    temp_dir
    ├── ood_detection
    │   └── project_name
    │       ├── data
    │       │   ├── Geti_dataset_1
    │       │   │   ├── images
    │       │   │   └── annotations
    │       │   ├── geti_dataset_2
    │       │   │   ├── images
    │       │   │   └── annotations
    │       │   ├── ood_images
    │       │   │   ├── image_001.jpg
    │       │   │   ├── image_002.jpg
    │       │       └── image_003.jpg




    """

    def __init__(
        self,
        geti: Geti,
        project: Union[str, Project],
        deployment: Deployment = None,
    ):
        """
        Model for Combined Out-of-Distribution (COOD) detection .
        :param geti: Geti instance representing the GETi server from which the project is to be used.
        :param project: Project or project name to use to fetch the deployment and the in-distribution data.
        The project must exist on the specified Geti instance and should have at least one trained model.
        :param deployment: Deployment to use for learning the data distribution. If None, a deployment with an XAI head is
        automatically selected from the project. If this COODModel is used in an OODTrigger,then make sure that
        the same deployment is used for post-inference hook.
        """
        self.geti = geti

        self.id_distribution_data = List[DistributionDataItem]
        self.ood_distribution_data = List[DistributionDataItem]

        self.ood_classifier = None  # The COOD random forest classifier

        if isinstance(project, str):
            project_name = project
            self.project = geti.get_project(project_name=project_name)
        else:
            self.project = project

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

        self.workspace_dir = os.path.join(
            tempfile.mkdtemp(), "ood_detection", self.project.name
        )
        self.data_dir = os.path.join(self.workspace_dir, "data")

        # Checks if project is a single-task classification project
        self._check_project_fit()

        if deployment is None:
            # If no deployment is provided, select an XAI model with the highest accuracy to be deployed
            self.deployment = get_usable_deployment(
                geti=self.geti,
                model_client=self.model_client,
            )
        else:
            if not deployment.models[0].has_xai_head:
                raise ValueError(
                    "The provided deployment does not have an model with an XAI head."
                    "Please reconfigure the deployment to include a model with an XAI head "
                    "(OptimizedModel.has_xai_head must be True). "
                    "Hint : You can use the get_usable_deployment() method from detect_ood.utils"
                )

            self.deployment = deployment

        if not self.deployment.are_models_loaded:
            self.deployment.load_inference_models(device="CPU")

        logging.info(
            f"Building Combined OOD detection model for Intel® Geti™ project `{self.project.name}`."
        )

        # The transformation to apply to in-distribution images to make them near-OOD
        self.corruption_transform = CutoutTransform()

        distribution_data = self._prepare_id_ood_data()
        self.id_distribution_data = distribution_data["id_data"]
        self.ood_distribution_data = distribution_data["ood_data"]

        # A dict consisting smaller OOD models (FRE, EnWeDi, etc)
        self.sub_models = {
            "knn_based": KNNBasedOODModel(knn_k=10),
            "class_fre": ClassFREBasedModel(n_components=0.995),
            "global_fre": GlobalFREBasedModel(n_components=0.995),
            "max_softmax_probability": ProbabilityBasedModel(),
        }

        self._train_sub_models()
        self.train()

    def _prepare_id_ood_data(self):
        datasets_in_project = self.project.datasets

        id_datasets_in_geti = []
        ood_datasets_in_geti = []

        # Figure out the datasets that can be used as a reference for in-distribution and out-of-distribution data
        for dataset in datasets_in_project:
            if dataset.name in ID_DATASET_NAMES:
                id_datasets_in_geti.append(dataset)
            elif dataset.name in OOD_DATASET_NAMES:
                ood_datasets_in_geti.append(dataset)

        if len(id_datasets_in_geti) == 0:
            raise ValueError(
                "Could not find any relevant datasets for in-distribution data. "
                "Please make sure that the project contains at least one dataset with the names: "
                f"{ID_DATASET_NAMES}."
            )

        # Prepare a List[DistributionDataItem] for in-distribution and out-of-distribution data

        id_distribution_data_items = []  # List[DistributionDataItem]
        for dataset in id_datasets_in_geti:
            id_distribution_data_items.extend(
                self._prepare_distribution_data(source=dataset)
            )

        ood_distribution_data_items = []  # List[DistributionDataItem]
        if len(ood_datasets_in_geti) == 0:
            logging.info(
                "No out-of-distribution datasets found in the project. "
                "Generating near-OOD images by applying strong corruptions to the in-distribution images."
            )
            for dataset in id_datasets_in_geti:
                ood_path = self._create_ood_images(reference_dataset=dataset)
                ood_distribution_data_items.extend(
                    self._prepare_distribution_data(source=ood_path)
                )

        else:
            for dataset in ood_datasets_in_geti:
                ood_distribution_data_items.extend(
                    self._prepare_distribution_data(source=dataset)
                )

        logging.info(
            f"Number of in-distribution samples: {len(id_distribution_data_items)}"
        )
        logging.info(
            f"Number of out-of-distribution samples: {len(ood_distribution_data_items)}"
        )

        return {
            "id_data": id_distribution_data_items,
            "ood_data": ood_distribution_data_items,
        }

    def _train_sub_models(self):
        """
        Initialise the OOD models
        """
        for sub_model in self.sub_models.values():
            sub_model.train(distribution_data=self.id_distribution_data)

    def _get_scores_from_sub_models(
        self, distribution_data: List[DistributionDataItem]
    ) -> dict:
        scores_all_sub_models = {}
        for ood_sub_model in self.sub_models:
            scores_dict = self.sub_models[ood_sub_model](distribution_data)
            for score_type in scores_dict:
                scores_all_sub_models[score_type] = scores_dict[score_type]
        return scores_all_sub_models

    def train(self):
        """
        Train the COOD model using the RandomForestClassifier
        """
        num_id_images = len(self.id_distribution_data)
        num_ood_images = len(self.ood_distribution_data)

        # Get scores from sub-models
        id_scores_all_sub_models = self._get_scores_from_sub_models(
            self.id_distribution_data
        )
        ood_scores_all_sub_models = self._get_scores_from_sub_models(
            self.ood_distribution_data
        )

        # Arrange features
        id_features = self._arrange_features(id_scores_all_sub_models, num_id_images)
        ood_features = self._arrange_features(ood_scores_all_sub_models, num_ood_images)

        # Combine features and labels
        all_features = np.concatenate((id_features, ood_features))
        # We take ood images as True or 1 and id images as False or 0
        all_labels = np.concatenate((np.zeros(num_id_images), np.ones(num_ood_images)))

        # Train the RandomForestClassifier
        self.ood_classifier = RandomForestClassifier()
        self.ood_classifier.fit(all_features, all_labels)

    def __call__(self, prediction: Prediction) -> float:
        """
        Return the COOD Score based using feature vector and prediction probabilities in "prediction".
        """
        data_item = DistributionDataItem(
            media_name="sample",  # We do not need this data for inference
            media_path="sample",
            annotated_label="",
            raw_prediction=prediction,
        )
        scores_all_sub_models = self._get_scores_from_sub_models(
            distribution_data=[data_item]
        )
        features_arranged = self._arrange_features(
            scores_all_sub_models=scores_all_sub_models, num_images=1
        )

        cood_score = self.ood_classifier.predict_proba(features_arranged)

        return cood_score[0][1]  # Return only the probability of being OOD

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
            corrupted_img = self.corruption_transform(img)
            corrupted_image_path = os.path.join(corrupted_images_path, image_name)
            cv2.imwrite(corrupted_image_path, corrupted_img)

        return corrupted_images_path

    def _check_project_fit(self):

        tasks_in_project = self.project.get_trainable_tasks()
        if len(tasks_in_project) != 1:
            raise ValueError(
                "Out-of-distribution detection is only "
                "supported for projects with a single task for now."
            )

        # get the task type and check if it is classification
        task_type = tasks_in_project[0].task_type
        if task_type != TaskType.CLASSIFICATION:
            raise ValueError(
                "Out-of-distribution detection models are only "
                "supported for classification tasks for now."
            )

    def _prepare_distribution_data(
        self, source: Union[Dataset, str]
    ) -> List[DistributionDataItem]:
        """
        Prepare the distribution data from the source by inferencing the images and extracting the feature vectors.
        param source: Dataset or directory containing images. If a dataset is provided, the images and annotations are
        downloaded from the dataset. If a directory is provided, the images are read from the directory.
        """
        if isinstance(source, Dataset):

            dataset_dir = os.path.join(self.data_dir, source.name)
            media_list = self.image_client.get_all_images(dataset=source)

            self.image_client.download_all(
                path_to_folder=dataset_dir, append_image_uid=True
            )
            self.annotation_client.download_annotations_for_images(
                images=media_list,
                path_to_folder=dataset_dir,
                append_image_uid=True,
            )

            annotations_dir = os.path.join(dataset_dir, "annotations")
            image_dir = os.path.join(dataset_dir, "images")

            image_paths = [
                os.path.join(image_dir, f"{media.name}_{media.id}.jpg")
                for media in media_list
            ]
            annotation_files = [
                os.path.join(annotations_dir, f"{media.name}_{media.id}.json")
                for media in media_list
            ]
        else:
            image_paths = [
                os.path.join(source, file_name) for file_name in os.listdir(source)
            ]
            annotation_files = [None] * len(image_paths)

        distribution_data_items = []
        for image_path, annotation_file in zip(image_paths, annotation_files):
            annotation_label = (
                self._load_annotations(annotation_file) if annotation_file else None
            )
            data_item = self._prepare_data_item(
                image_path=image_path, annotation_label=annotation_label
            )
            distribution_data_items.append(data_item)

        return distribution_data_items

    def _prepare_data_item(
        self, image_path: str, annotation_label: Union[str, None]
    ) -> DistributionDataItem:
        prediction = self._infer(image_path=image_path, explain=True)
        return DistributionDataItem(
            media_name=os.path.splitext(os.path.basename(image_path))[0],
            media_path=image_path,
            annotated_label=annotation_label,
            raw_prediction=prediction,
        )

    @staticmethod
    def _load_annotations(annotation_file: str) -> Union[str, None]:
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                annotation = json.load(f)
                return annotation["annotations"][0]["labels"][0]["name"]
        return None

    @staticmethod
    def _arrange_features(scores_all_sub_models: dict, num_images: int) -> np.ndarray:
        features = np.zeros((num_images, len(scores_all_sub_models)))
        for score_idx, score_type in enumerate(scores_all_sub_models):
            features[:, score_idx] = scores_all_sub_models[score_type]
        return features


class OODSubModel(metaclass=ABCMeta):
    """
    Base class for OOD detection sub-models.
    """

    def __init__(self):
        self._is_trained = False

    @abstractmethod
    def train(self, distribution_data: List[DistributionDataItem]):
        """
        Train the OOD detection sub-model using a list in-distribution data items
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, data_items: List[DistributionDataItem]) -> dict:
        """
        Return the OOD score for the given data items.
        """
        raise NotImplementedError

    @property
    def is_trained(self) -> bool:
        """
        Return True if the model is trained.
        """
        return self._is_trained


class KNNBasedOODModel(OODSubModel):
    """
    k Nearest Neighbour based OOD detection model.
    # TODO[OOD]: Add
    1) distance to prototypical center
    2) ldof (to expensive ?)
    3) exact combination of entropy and distance from thesis

    """

    def __init__(self, knn_k: int = 10):
        super().__init__()
        self.knn_k = knn_k
        self.knn_search_index = None
        self.train_set_labels = None
        self._is_trained = False

    def train(self, distribution_data: List[DistributionDataItem]):
        """
        Train the kNN model
        """
        id_data = distribution_data
        feature_vectors = np.array([data.feature_vector for data in id_data])
        labeled_set_labels = np.array([data.annotated_label for data in id_data])

        self.train_set_labels = labeled_set_labels
        self.knn_search_index = perform_knn_indexing(feature_vectors, use_gpu=False)
        self._is_trained = True

    def __call__(self, data_items: List[DistributionDataItem]) -> dict:
        """
        Return the kNN OOD score for the given feature vector.
        """
        if not self._is_trained:
            raise ValueError(
                "Model is not trained. Please train the model first before calling."
            )
        features = np.array([item.feature_vector for item in data_items])
        distances, nn_indices = perform_knn_search(
            knn_search_index=self.knn_search_index,
            feature_vectors=features,
            k=self.knn_k,
        )

        knn_distance = distances[:, -1]  # distance to the kth nearest neighbour
        nn_distance = distances[:, 1]  # distance to the nearest neighbour
        # TODO : When doing kNN Search for ID, the 0th index is the same image. So, should we use k+1 ?
        average_nn_distance = np.mean(distances[:, 1:], axis=1)

        entropy_score = calculate_entropy_nearest_neighbours(
            train_labels=self.train_set_labels,
            nns_labels_for_test_fts=nn_indices,
            k=self.knn_k,
        )

        # Add one to the entropy scores
        # This is to offset the range to [1,2] instead of [0,1] and avoids division by zero
        # if used elsewhere
        entropy_score += 1

        enwedi_score = average_nn_distance * entropy_score
        enwedi_nn_score = nn_distance * entropy_score

        return {
            "knn_distance": knn_distance,
            "nn_distance": nn_distance,
            "average_nn_distance": average_nn_distance,
            "entropy_score": entropy_score,
            "enwedi_score": enwedi_score,
            "enwedi_nn_score": enwedi_nn_score,
        }


class GlobalFREBasedModel(OODSubModel):
    """
    Yet to be finalised
    """

    def __init__(self, n_components=0.995):
        super().__init__()
        self.n_components = n_components
        self.pca_model = None
        self._is_trained = False

    def train(self, distribution_data: List[DistributionDataItem]):
        """
        Trains a single (global) PCA model for the in-distribution data
        """
        feature_vectors = np.array([data.feature_vector for data in distribution_data])
        self.pca_model = fit_pca_model(
            feature_vectors=feature_vectors, n_components=self.n_components
        )
        self._is_trained = True

    def __call__(self, data_items: List[DistributionDataItem]) -> dict:
        """
        Return the global fre score for the given data items.
        """
        if not self._is_trained:
            raise ValueError(
                "Model is not trained. Please train the model first before calling."
            )
        features = np.array([item.feature_vector for item in data_items])
        fre_scores = fre_score(feature_vectors=features, pca_model=self.pca_model)
        return {"global_fre_score": fre_scores}


class ClassFREBasedModel(OODSubModel):
    """
    Yet to be finalised
    """

    def __init__(self, n_components=0.995):
        super().__init__()
        self.n_components = n_components
        self.pca_models_per_class = {}
        self._is_trained = False

    def train(self, distribution_data: List[DistributionDataItem]):
        """
        Fit PCA Models on the in-distribution data for each class.
        """
        id_data = distribution_data
        feature_vectors = np.array([data.feature_vector for data in id_data])
        labels = np.array([data.annotated_label for data in id_data])

        # iterate through unique labels and fit pca model for each class
        pca_models = {}

        for label in np.unique(labels):
            # labels are list of class names and not indices
            class_indices = [i for i, j in enumerate(labels) if j == label]
            class_features = feature_vectors[class_indices]
            pca_models[label] = fit_pca_model(
                feature_vectors=class_features, n_components=self.n_components
            )

        self.pca_models_per_class = pca_models
        self._is_trained = True

    def __call__(self, data_items: List[DistributionDataItem]) -> dict:
        """
        Return the class fre score for the given feature vector.
        """
        if not self._is_trained:
            raise ValueError(
                "Model is not trained. Please train the model first before calling."
            )
        features = np.array([item.feature_vector for item in data_items])
        fre_scores_per_class = {}
        # class_fre_models is a dict with label name and pca model.
        for label, pca_model in self.pca_models_per_class.items():
            fre_scores_per_class[label] = fre_score(
                feature_vectors=features, pca_model=pca_model
            )

        # For each element, we return the max FRE score
        max_fre_scores = np.ndarray(len(features))
        for k in range(len(features)):
            max_fre_scores[k] = max(
                [fre_scores_per_class[label][k] for label in fre_scores_per_class]
            )
        return {"class_fre_score": max_fre_scores}


class ProbabilityBasedModel(OODSubModel):
    """
    Maximum Softmax Probability Model - A baseline OOD detection model.
    Use the concept that a lower maximum softmax probability indicates that the image could be OOD.
    See
    """

    def __init__(self):
        super().__init__()
        self._is_trained = False

    def train(self, distribution_data: List[DistributionDataItem]):
        """
        MSP model does not require training.
        """
        self._is_trained = True

    def __call__(self, data_items: List[DistributionDataItem]) -> dict:
        """
        Return the maximum softmax probability for the given prediction.
        """
        if not self._is_trained:
            raise ValueError(
                "Model is not trained. Please train the model first before calling."
            )
        msp_scores = np.ndarray(len(data_items))
        for i, data_item in enumerate(data_items):
            msp_scores[i] = data_item.max_prediction_probability[0]

        return {"max_softmax_probability": msp_scores}
