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


import numpy as np
from sklearn.decomposition import PCA

from geti_sdk.deployment import Deployment
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_clients import ImageClient


def fit_pca_model(representation_vectors=np.ndarray, n_components: float = 0.995):
    """
    Fit a Principal component analysis (PCA) model to the features and returns the model
    :param representation_vectors: Train set features to fit the PCA model
    :param n_components: Number of components (fraction of variance) to keep
    """
    pca_model = PCA(n_components)
    pca_model.fit(representation_vectors)
    return pca_model


def fre_score(representation_vectors: np.ndarray, pca_model: PCA) -> np.ndarray:
    """
    Return the feature reconstruction error (FRE) score for a given feature vector(s)
    :param representation_vectors: feature vectors to compute the FRE score
    :param pca_model: PCA model to use for computing the FRE score. PCA model must be fitted already
    """
    features_original = representation_vectors
    features_transformed = pca_model.transform(representation_vectors)
    features_reconstructed = pca_model.inverse_transform(features_transformed)
    fre_scores = np.sum(np.square(features_original - features_reconstructed), axis=1)
    return fre_scores


def extract_features_from_imageclient(
    deployment: Deployment,
    image_client: ImageClient,
    geti_session: GetiSession,
    n_images: int = -1,
    normalise_feats: bool = True,
):
    """
    Extract
    """
    pass


def generate_ood_dataset_by_corruption(
    geti_deployment: Deployment,
    source_path: str,
    corruption_type: str,
    dest_path: str = None,
    desired_accuracy: float = 50,
    desired_accuracy_tol=3.0,
    show_progress: bool = True,
) -> str:
    """
    Util
    """
    pass
