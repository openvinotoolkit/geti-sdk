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


from .image import (
    get_grid_arrangement,
    get_image_paths,
    calc_classification_accuracy
)
from .augmentations import TransformImages

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from geti_sdk.deployment import Deployment
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_clients import ImageClient


def show_top_n_misclassifications(
        images_dir: str,
        scores: np.ndarray,
        type_of_samples: str,
        n_images: int = 9,
):
    """
    Show top n misclassified images based on their OOD scores
    (sorted by score in descending order for in-distribution samples and in ascending order for OOD samples).
    :param images_dir: Path to directory with images.
    :param scores: OOD scores for images. For in-distribution samples, the lower the score, the more OOD the sample is.
    :param type_of_samples: Type of samples to show. Must be one of ['id', 'ood'].
    :param n_images: Number of images to show.

    """
    images_paths_and_labels = get_image_paths(images_dir)
    if type_of_samples == "id":
        score_sort_inds = np.argsort(scores)
    elif type_of_samples == "ood":
        score_sort_inds = np.argsort(scores)[::-1]
    else:
        raise ValueError(
            f"type_of_samples must be one of ['id', 'ood'], got {type_of_samples}"
        )

    images_paths = list(images_paths_and_labels.keys())
    image_paths_sorted_by_score = [images_paths[k] for k in score_sort_inds]

    n_rows, n_cols = get_grid_arrangement(n_images)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    for i, ax in enumerate(axes.flatten()):
        image = cv2.imread(image_paths_sorted_by_score[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480, 480))
        ax.imshow(image)
        label = images_paths_and_labels[image_paths_sorted_by_score[i]]
        ax.set_title(
            f"Label: {label} \n (Sore : {scores[score_sort_inds[i]]:.2f})",
            color="#0068b5",
            fontsize=11,
            wrap=True,
        )
        ax.axis("off")
    fig.suptitle(f"Top {n_images} misclassified {type_of_samples} images", fontsize=16)
    plt.tight_layout()
    plt.show()


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
    Generate a dataset of corrupted images from a source dataset of clean images.
    The corruption is applied until the classification accuracy (tp/tp+fp) on the generated dataset reached
    the desired accuracy.
    :geti_deployment: The trained geti deployment (model) to use for classification.
    :source_path: The path to the source dataset of clean images. The dataset is required to have the following structure:
        source_path/
            class_1/
                image_1.jpg
                image_2.jpg
                ...
            class_2/
                image_1.jpg
                image_2.jpg
                ...
            ...
    :corruption_to_apply: The type of corruption to apply. Currently supported corruptions
        are: self._SUPPORTED_CORRUPTIONS
    :desired_accuracy: The desired classification accuracy in percentage on the generated dataset.
        A 50% accuracy means that the model is not able to correctly classify half of the
        images in the dataset.
    :show_progress: If True, a progress bar will be displayed.
    :return: The path to the generated dataset.
    """
    print(
        f"Generating OOD dataset by applying {corruption_type} corruption on {source_path}"
    )

    transform_images = TransformImages(corruption_type=corruption_type)
    dataset_folder_name = os.path.basename(source_path)
    if dest_path is None:
        dest_path = os.path.join(
            os.path.dirname(source_path),
            f"{dataset_folder_name}_{transform_images.corruption_type}_{desired_accuracy:.0f}",
        )
    accuracy = calc_classification_accuracy(
        dataset_path=source_path,
        deployment=geti_deployment,
        show_progress=show_progress,
    )
    if accuracy < desired_accuracy:
        print(f"Maximum possible accuracy : {accuracy:.2f} %")
        print(f"Can not reach desired accuracy of {desired_accuracy:.2f} %")
        return source_path
    if show_progress:
        print(f"Accuracy without any corruptions applied : {accuracy:.2f} %")
    corruption_strength = transform_images.corruption_strength_range[0]
    while abs(accuracy - desired_accuracy) > desired_accuracy_tol:
        corruption_strength = transform_images.update_corruption_strength(desired_accuracy=desired_accuracy,
                                                                          current_accuracy=accuracy,
                                                                          current_strength=corruption_strength)
        transform_images.apply_corruption_on_folder(
            source_path=source_path,
            dest_path=dest_path,
            corruption_strength=corruption_strength,
            show_progress=show_progress,
        )
        accuracy = calc_classification_accuracy(
            dataset_path=dest_path,
            deployment=geti_deployment,
            show_progress=show_progress,
        )
        if show_progress:
            print(f"Current accuracy: {accuracy:.2f}")
    if abs(accuracy - desired_accuracy) < desired_accuracy_tol:
        print(f"Corrupted dataset generated with accuracy {accuracy:.2f} %")
    return dest_path


def extract_features_from_imageclient(
        deployment: Deployment,
        image_client: ImageClient,
        geti_session: GetiSession,
        n_images: int = -1,
        normalise_feats: bool = True,
):
    """
    Extract feature embeddings from a Geti deployment model for a given number of images in a geti image_client
    :param deployment: The trained Geti deployment (model) to use for feature extraction.
    :param image_client: The Geti ImageClient object containing the images to extract features from.
    :param geti_session: The GetiSession object.
    :param n_images: The number of images to extract features from.
        If -1, all images in the image_client will be used. Else a random will be picked.
    :param normalise_feats: If True, the feature embeddings are normalised by dividing each feature
        embedding vector by its respective 2nd-order vector norm (vector Euclidean norm)
    :return: A numpy array containing the extracted feature embeddings of shape (n_images, feature_len)
    """
    print("Retrieving the list of images from the project ...")
    images_in_client = image_client.get_all_images()
    total_n_images = len(images_in_client)  # total number of images in the project
    if n_images == -1:
        n_images = total_n_images

    n_images = min(n_images, total_n_images)
    sample_image = images_in_client[0].get_data(session=geti_session)
    prediction = deployment.explain(sample_image)
    feature_len = prediction.feature_vector.shape[0]
    # pick random images - Stratified sampling not possible yet as we don't have labels in image_client
    random_indices = np.random.choice(total_n_images, n_images, replace=False)
    features = np.zeros((n_images, feature_len))
    for i, k in tqdm(
            enumerate(random_indices), total=n_images, desc="Extracting features"
    ):
        image_numpy = images_in_client[k].get_data(session=geti_session)
        prediction = deployment.explain(image_numpy)
        features[i] = prediction.feature_vector

    if normalise_feats:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

    return features
