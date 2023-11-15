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

import os
from math import sqrt
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image as PILImage
from tqdm import tqdm

from geti_sdk.data_models import Image
from geti_sdk.deployment import Deployment


def simulate_low_light_image(
    image: Union[np.ndarray, Image], reduction_factor: float = 0.5
) -> np.ndarray:
    """
    Simulate a reduced intensity and exposure time for an input image.
    It does so by reducing the image brightness and adding simulated shot noise to the
    image.

    :param image: Original image
    :param reduction_factor: Brightness reduction factor. Should be in the
        interval [0, 1]
    :return: Copy of the image, simulated for low lighting conditions (lower
        intensity and exposure time).
    """
    if isinstance(image, np.ndarray):
        new_image = image.copy()
    elif isinstance(image, Image):
        new_image = image.numpy.copy()
    else:
        raise TypeError(f"Unsupported image type '{type(image)}'")

    # Reduce brightness
    new_image = cv2.convertScaleAbs(new_image, alpha=reduction_factor, beta=0)

    # Add some shot noise to simulate reduced exposure time
    PEAK = 255 * reduction_factor
    new_image_with_noise = np.clip(
        np.random.poisson(new_image / 255.0 * PEAK) / PEAK * 255, 0, 255
    ).astype("uint8")
    return new_image_with_noise


def display_image_in_notebook(image: Union[np.ndarray, Image], bgr: bool = True):
    """
    Display an image inline in a Juypter notebook.

    :param image: Image to display
    :param bgr: True if the image has its channels in BGR order, False if it
        is in RGB order. Defaults to True
    """
    new_image = image_to_np(image)
    if bgr:
        result = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    else:
        result = new_image
    img = PILImage.fromarray(result)
    display(img)


def image_to_np(
    image: Union[np.ndarray, Image],
) -> np.ndarray:
    """
    Make sure an image is a numpy array.
    """
    if isinstance(image, np.ndarray):
        new_image = image.copy()
    elif isinstance(image, Image):
        new_image = image.numpy.copy()
    else:
        raise TypeError(f"Unsupported image type '{type(image)}'")
    return new_image


def calc_classification_accuracy(
    dataset_path: str,
    deployment: Deployment,
    show_progress: bool = True,
) -> float:
    """
    Calculate the classification accuracy of a dataset using a trained Geti deployment.
    :param dataset_path: The path to the (test) dataset. The dataset is supposed to have the following structure:
        dataset_path/
            class_1/
                image_1.jpg
                image_2.jpg
                ...
            class_2/
                image_1.jpg
                image_2.jpg
                ...
            ...
    :param deployment: The trained Geti deployment (model) to use for classification.
    :param show_progress: If True, a progress bar will be displayed.
    :return: The classification accuracy of the dataset in percentage.
    """
    id_images_dict = get_image_paths(src_dir=dataset_path, images_dict=None, label=None)
    correct_classifications = 0
    for image_path, label in tqdm(
        id_images_dict.items(), disable=not show_progress, desc="Calculating Accuracy"
    ):
        numpy_image = cv2.imread(image_path)
        numpy_rgb = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        # numpy_rgb = cv2.resize(numpy_rgb, (480, 480))
        prediction = deployment.infer(numpy_rgb)
        # pred_prob = prediction.annotations[0].labels[0].probability
        pred_label = prediction.annotations[0].labels[0].name
        if pred_label == label:
            correct_classifications += 1
    accuracy = correct_classifications / len(id_images_dict)
    return accuracy * 100


def get_image_paths(src_dir, images_dict=None, label=None):
    """
    Recursively retrieves the paths of all images in a directory.
    :param src_dir: The path to the directory containing the images.
    :param images_dict: A dictionary containing the image paths and their corresponding labels.
        If not None, the retrieved image paths will be added to this dictionary.
    :param label: The label to assign to the images in the directory.
        If None, the directory name will be used as label.
    """
    if images_dict is None:
        images_dict = {}
    if label is None:
        label = os.path.basename(src_dir)
    for img_file in os.listdir(src_dir):
        img_path = os.path.join(src_dir, img_file)
        if os.path.isdir(img_path):
            get_image_paths(img_path, images_dict, label=None)
        else:
            images_dict[img_path] = label
    return images_dict


def extract_features_from_img_folder(
    deployment: Deployment, images_folder_path: str, normalise_feats: bool = True
):
    """
    Extract feature embeddings from a Geti deployment model for images in a folder
    :param deployment: The trained Geti deployment (model) to use for feature extraction.
    :param images_folder_path: The path to the folder containing the images to extract features from.
    :param normalise_feats: If True, the feature embeddings are normalised by dividing each feature
        embedding vector by its respective 2nd-order vector norm (vector Euclidean norm)
    :return: A numpy array containing the extracted feature embeddings of shape (n_images, feature_len)
    """
    if not os.path.isdir(images_folder_path):
        raise ValueError(f"img_folder {images_folder_path} is not a valid directory")

    images_in_folder = get_image_paths(images_folder_path)
    sample_image = cv2.imread(list(images_in_folder.keys())[0])
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    prediction = deployment.explain(sample_image)
    feature_len = prediction.feature_vector.shape[0]

    features = np.zeros((len(images_in_folder.keys()), feature_len))

    for k, id_image_path in tqdm(
        enumerate(images_in_folder.keys()),
        total=len(images_in_folder.keys()),
        desc="Extracting features",
    ):
        numpy_image = cv2.imread(id_image_path)
        numpy_rgb = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        prediction = deployment.explain(numpy_rgb)
        features[k] = prediction.feature_vector

    if normalise_feats:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

    return features


def get_grid_arrangement(n: int):
    """
    Find the number of rows and columns for a grid arrangement of n items in a grid.
    This function returns the grid arrangement with the closest number of rows and columns to the square root of n.
    """
    factors = []
    for current_factor in range(n):
        if n % float(current_factor + 1) == 0:
            factors.append(current_factor + 1)

    index_closest_to_sqrt = min(
        range(len(factors)), key=lambda i: abs(factors[i] - sqrt(n))
    )

    if factors[index_closest_to_sqrt] * factors[index_closest_to_sqrt] == n:
        return factors[index_closest_to_sqrt], factors[index_closest_to_sqrt]
    else:
        index_next = index_closest_to_sqrt + 1
        return factors[index_closest_to_sqrt], factors[index_next]


def display_sample_images_in_folder(
    images_path: str, title: str = None, n_images: int = 9, show_labels: bool = True
):
    """
    Display a random sample of images from a dataset folder
    :param images_path: path to the folder containing the images
    :param title: title of the plot
    :param n_images: number of images to display
    :param show_labels: whether to show the name of the subfolder of the image as its title
    """
    if not os.path.isdir(images_path):
        raise ValueError(f"images_path {images_path} is not a valid directory")

    images_in_folder = get_image_paths(images_path)
    random_indices = np.random.choice(
        len(images_in_folder.keys()), n_images, replace=False
    )
    n_rows, n_cols = get_grid_arrangement(n_images)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for i, ax in enumerate(axes.flatten()):
        image = cv2.imread(list(images_in_folder.keys())[random_indices[i]])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480, 480))
        ax.imshow(image)
        if show_labels:
            ax.set_title(
                list(images_in_folder.values())[random_indices[i]],
                color="#0068b5",
                fontsize=11,
                wrap=True,
            )
        ax.axis("off")
    if title is None:
        title = f"Sample images from {images_path}"
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
