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

from typing import List

import albumentations
import faiss
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from geti_sdk import Geti
from geti_sdk.deployment import Deployment
from geti_sdk.rest_clients import ModelClient


def get_deployment_with_xai_head(geti: Geti, model_client: ModelClient) -> Deployment:
    """
    Get a deployment that has an optimised model with an XAI head. If there are multiple models with XAI heads,
    the model with the best performance is selected.
    :param geti: Geti instance pointing to the GETi server
    :param model_client: Modelclient instance pointing to the Project where at least one trained model is present
    :return: Deployment object with the optimised model with an XAI head.
    """
    # Check if there's at least one trained model in the project
    models = model_client.get_all_active_models()
    if len(models) == 0:
        raise ValueError(
            "No trained models were found in the project, please either "
            "train a model first or specify an algorithm to train."
        )

    # We need the model which has xai enabled - this allows us to get the feature vector from the model.
    model_with_xai_head = None

    # TODO[OOD] : More model properties can be used to determine "best" model (size, precision with respect to accuracy)
    max_model_performance = -1
    for model in models:
        for optimised_model in model.optimized_models:
            if optimised_model.has_xai_head:
                model_performance = optimised_model.performance.score
                if model_performance > max_model_performance:
                    model_with_xai_head = optimised_model
                    max_model_performance = model_performance

    if model_with_xai_head is None:
        raise ValueError(
            "No trained model with an XAI head was found in the project, "
            "please train a model with an XAI head first."
        )

    deployment = geti.deploy_project(
        project_name=model_client.project.name, models=[model_with_xai_head]
    )

    return deployment


def fit_pca_model(feature_vectors=np.ndarray, n_components: float = 0.995) -> PCA:
    """
    Fit a Principal component analysis (PCA) model to the features and returns the model
    :param feature_vectors: Train set features to fit the PCA model
    :param n_components: Number of components (fraction of variance) to keep
    :return: A fitted PCA model
    """
    pca_model = PCA(n_components)
    pca_model.fit(feature_vectors)
    return pca_model


def stratified_selection(
    x, y, fraction: float, min_samples_per_class: int = 3
) -> (List, List):
    """
    Sub sample (reduce) a dataset (x,y) by a provided fraction while maintaining the class distribution
    Note that this is to be use only for collection where each x (data point or sample) has only one y (label).

    :param x: Data points (samples)
    :param y: Labels
    :param fraction: Fraction of the dataset to keep.
    :param min_samples_per_class: Minimum number of samples to keep per class. Note that a very small value for
    "fraction" can sometimes make a class empty. To avoid this, we keep a minimum number of samples per class.
    :return: Indices of the data points to keep in the reduced split
    """
    selected_data_indices = []

    samples = x
    labels = y

    # Check if labels is empty
    if len(labels) == 0:
        raise ValueError("Labels cannot be empty")

    # Check if len of labels and samples are equal
    if len(labels) != len(samples):
        raise ValueError("Length of labels and samples must be equal")

    if type(labels) is list:
        labels = np.array(labels)

    # Get unique labels
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        # Get number of samples to keep
        n_samples_to_keep = max(
            min_samples_per_class, int(fraction * len(label_indices))
        )
        selected_indices = np.random.choice(
            label_indices, n_samples_to_keep, replace=False
        )
        # Append selected samples and labels
        selected_data_indices.extend(selected_indices)

    return selected_data_indices


def fre_score(feature_vectors: np.ndarray, pca_model: PCA) -> np.ndarray:
    """
    Calculate the feature reconstruction error (FRE) score for the given feature vector(s)
    :param feature_vectors: feature vectors to compute the FRE score
    :param pca_model: PCA model to use for computing the FRE score. PCA model must be fitted already
    :return: FRE scores for the given feature vectors
    """
    features_original = feature_vectors
    features_transformed = pca_model.transform(feature_vectors)
    features_reconstructed = pca_model.inverse_transform(features_transformed)
    fre_scores = np.sum(np.square(features_original - features_reconstructed), axis=1)
    return fre_scores


def perform_knn_indexing(
    feature_vectors: np.ndarray, use_gpu: bool = False
) -> faiss.IndexFlatL2:
    """
    Perform KNN indexing on the feature vectors using the FAISS library
    :param feature_vectors: Feature vectors to build the KNN index on
    :param use_gpu: Whether to use GPU for KNN indexing. Default is False
    :return: KNN search index object
    """
    # use faiss with gpu
    if use_gpu:
        res = faiss.StandardGpuResources()
        # build a flat (CPU) index
        index_flat = faiss.IndexFlatL2(feature_vectors.shape[1])
        # make it into a gpu index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index_flat.add(feature_vectors)
        return gpu_index_flat
    else:
        index_flat = faiss.IndexFlatL2(feature_vectors.shape[1])
        index_flat.add(feature_vectors)
        return index_flat


def perform_knn_search(
    knn_search_index: faiss.IndexFlatL2, feature_vectors: np.ndarray, k: int = 10
) -> (np.ndarray, np.ndarray):
    """
    Perform KNN search on the feature vectors in the feature space indexed by the knn_search_index
    :param knn_search_index: KNN search index. An object representing the indexed knn search space.
    Ideally this object is returned by perform_knn_indexing().
    :param feature_vectors: Query feature vectors to search in the indexed feature space.
        Note that the feature_vectors' size should be (N, M) where N is the number of feature vectors
        and M is the dimension of the feature vectors.
    :param k: Number of nearest neighbours to search for
    :return: distances, indices each of size (N,K). Note that distances are squared Euclidean distances.
    """
    distances, indices = knn_search_index.search(feature_vectors, k)

    return distances, indices


def calculate_entropy_nearest_neighbours(
    train_labels: np.ndarray,
    nns_labels_for_test_fts: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Calculate the "entropy", a measure of how different the k nearest neighbours are for a sample.
    The value always range between [0,1] . A 0 indicates that all the k nearest neighbours belong to one class.
        Not a highly informative sample
    A value of 1 indicates that the sample has k different nearest neighbours (no sample belongs to same class).
        More informative sample in theory

    :param train_labels: labels of the annotated images
    :param nns_labels_for_test_fts: labels of the k nearest neighbours for each test feature
    :param k: number of nearest neighbours to consider
    :return: Entropy scores for each test feature
    """
    # preallocate
    neighbour_bin_count = np.zeros((nns_labels_for_test_fts.shape[0], k), dtype=int)
    for i in range(nns_labels_for_test_fts.shape[0]):
        nn_labels = train_labels[nns_labels_for_test_fts[i, :]]

        _, nn_bin_count = np.unique(
            nn_labels,
            return_index=False,
            return_inverse=False,
            return_counts=True,
        )
        neighbour_bin_count[i, : nn_bin_count.shape[0]] = nn_bin_count
        # No correction for all samples from a class being in a test sample's nearest neighbours
        # is done.

    # Calculate entropy
    # Note : This Entropy lies within [0,1]
    # A fully uncertain sample has entropy of 1 (bin count looks like [1,1,1,1,1,1,1,1,1,1])
    # A fully certain sample has entropy of 0 (bin count looks like [10,0,0,0,0,0,0,0,0,0])
    entropy_scores = stats.entropy(neighbour_bin_count, axis=1, base=k)
    return entropy_scores


def normalise_features(feature_vectors: np.ndarray) -> np.ndarray:
    """
    Feature embeddings are normalised by dividing each feature embedding vector by its respective 2nd-order vector norm
    (vector Euclidean norm). It has been shown that normalising feature embeddings lead to a significant improvement
    in OOD detection.
    :param feature_vectors: Feature vectors to normalise
    :return: Normalised feature vectors.
    """
    if len(feature_vectors.shape) == 1:
        feature_vectors = feature_vectors.reshape(1, -1)

    return feature_vectors / (
        np.linalg.norm(feature_vectors, axis=1, keepdims=True) + 1e-10
    )


class CutoutTransform:
    """
    Cutout transform to apply on images. This can be used for generating out of distribution (OOD) samples from in
    distribution (ID) samples.
    """

    def __init__(
        self,
        number_of_cutouts: int = 1,
        min_cutout_size: float = 0.5,
        max_cutout_size: float = 0.7,
    ):
        """
        :param number_of_cutouts: Number of cutouts to apply on the image
        :param min_cutout_size: Minimum size of the cutout
        :param max_cutout_size: Maximum size of the cutout
        """
        # TODO[OOD]: Add more advanced OOD transforms like perlin noise
        transform = albumentations.Compose(
            [
                albumentations.CoarseDropout(
                    max_holes=number_of_cutouts,
                    p=1,
                    hole_width_range=(min_cutout_size, max_cutout_size),
                    hole_height_range=(min_cutout_size, max_cutout_size),
                )
            ]
        )
        self.transform = transform

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the cutout transform on the image
        """
        return self.transform(image=image)["image"]
