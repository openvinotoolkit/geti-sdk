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

from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from geti_sdk.data_models import Prediction

from .utils import fit_pca_model, fre_score


class OODModel:
    """
    Out-of-distribution detection model.
    Uses the Combined out-of-distribution (COOD) detection
    algorithm (see : https://arxiv.org/abs/2403.06874).
    """

    def __init__(self):
        """
        Template
        """
        self.ood_classifier = None  # The COOD random forest classifier
        self.sub_models = None  # A dict consisting submodels (FRE, EnWeDi, MSP, etc)

        pass

    def __call__(self, prediction: Prediction) -> float:
        """
        Return the COOD Score based using feature vector and prediction probabilities in "prediction".
        """
        # feature_vector = prediction.feature_vector
        # for annotation in prediction.annotations:
        #     # Better way to get probability (or logits_)
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

    def initialise_sub_models(self):
        """
        Initialise all the sub-models (FRE, EnWeDi, etc). This is done before training the COOD model.
        """
        pass

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
        # Step 4 : Initialise/Index/Train all the sub-models (FRE,EnWeDi, etc)
        # Step 5 : Forward pass through sub-models to get ood scores for each image
        # Step 6 : Train the COOD Random Forest
        # Step 7 : Test COOD on test set (?)  Determine threshold (usually this is just 0.5)
        ood_classifier = RandomForestClassifier()
        features = []  # Each element is an output (ood score) from the sub-models
        labels = []  # OOD = 1, ID = 0
        ood_classifier.fit(features, labels)

        self.ood_classifier = ood_classifier

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


class ClassFREModel:
    """
    Yet to be finalised
    """

    def __init__(self):
        self.class_fre_models = None
        self.n_components = 0.995

    def __call__(self, features: np.ndarray, prediction: np.ndarray) -> float:
        """
        Return the class fre score for the given feature vector.
        """
        fre_scores_per_class = {}
        # class_fre_models is a dict with label name and pca model.
        for label, pca_model in self.class_fre_models.items():
            fre_scores_per_class[label] = fre_score(
                representation_vectors=features,
                pca_model=pca_model,
            )

        # return maximum FRE
        return max(fre_scores_per_class.values())

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
            pca_models[label] = fit_pca_model(
                train_features=features[class_indices],
                n_components=0.995,
            )

        self.class_fre_models = pca_models


class MaxSoftmaxProbabilityModel:
    """
    Maximum Softmax Probability Model - A baseline OOD detection model.
    Use the concept that a lower maximum softmax probability indicates that the image could be OOD.
    See
    """

    def __init__(self):
        pass


class EnWeDiModel:
    """
    Entropy Weighted Nearest Neighbour Model. Copy description from ILRF class
    """

    def __init__(self):
        pass
