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
import os
from math import sqrt
from typing import Union

import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image as PILImage
from tqdm import tqdm

from geti_sdk.data_models import Image
from geti_sdk.deployment import Deployment
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_clients import ImageClient


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
    new_image = _image_to_np(image)
    if bgr:
        result = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    else:
        result = new_image
    img = PILImage.fromarray(result)
    display(img)


def _image_to_np(
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


class TransformImages:
    """
    Class for applying corruptions/augmentations to images and datasets
    """

    def __init__(
        self,
        corruption_type: str = "motion_blur",
    ):
        """
        :param corruption_type: The type of corruption to apply.

        """
        self.corruption_type = corruption_type
        self._SEED = 117
        self._SUPPORTED_CORRUPTIONS = [
            "motion_blur",
            "gaussian_blur",
            "cut_out",
            "fake_snow",
            "poisson_noise",
        ]
        if self.corruption_type not in self._SUPPORTED_CORRUPTIONS:
            raise ValueError(
                f"Unsupported corruption type '{self.corruption_type}, "
                f"supported corruption types are {self._SUPPORTED_CORRUPTIONS}"
            )

        self._corruption_strength_range = (1, 300)
        if self.corruption_type in ["cut_out", "fake_snow"]:
            self._corruption_strength_range = (0.001, 1)
        elif self.corruption_type == "poisson_noise":
            self._corruption_strength_range = (
                1,
                110,
            )  # Increase this if the desired accuracy is not reached
        elif self.corruption_type == "gaussian_blur":
            self._corruption_strength_range = (
                1,
                8,
            )  # Increase this if the desired accuracy is not reached
        elif self.corruption_type == "motion_blur":
            self._corruption_strength_range = (
                1,
                110,
            )  # Increase this if the desired accuracy is not reached

    def _compose_imgaug_corruption(self, corruption_strength: Union[float, int]):
        transform = None
        if self.corruption_type == "motion_blur":
            transform = iaa.MotionBlur(
                k=int(round(corruption_strength)),
                angle=45,
                direction=0.5,
                seed=self._SEED,
            )

        elif self.corruption_type == "gaussian_blur":
            transform = iaa.GaussianBlur(sigma=corruption_strength, seed=self._SEED)

        elif self.corruption_type == "cut_out":
            transform = iaa.Cutout(
                nb_iterations=2,  # No. of boxes per image
                size=corruption_strength,
                squared=False,
                seed=self._SEED,
                position="uniform",  # "uniform" - random, "normal" - center of the image
                fill_mode="constant",
                cval=0,
            )  # fill with black boxes

        elif self.corruption_type == "fake_snow":
            # SnowFlakes from ImgAug can give error depending on the numpy version you have.
            transform = iaa.Snowflakes(
                density=0.075,
                density_uniformity=1.0,
                flake_size=corruption_strength,
                flake_size_uniformity=0.5,
                angle=45,
                speed=0.025,
                seed=self._SEED,
            )

        elif self.corruption_type == "poisson_noise":
            transform = iaa.AdditivePoissonNoise(
                lam=corruption_strength, per_channel=True, seed=self._SEED
            )

        return transform

    def _apply_corruption_on_image(
        self,
        image: Union[np.ndarray, Image],
        corruption_strength: Union[float, int] = 0.5,
    ) -> np.ndarray:
        """
        Apply a corruption to an image.

        :param image: Original image
        :param corruption_strength: The strength of the corruption. Ignored if corruption_type
        is an albumentations.Compose object.
        :return: Copy of the image with augmentation/corruption applied.
        """
        input_image = _image_to_np(image)
        transformation = self._compose_imgaug_corruption(
            corruption_strength=corruption_strength
        )
        transformed_image = transformation(image=input_image)
        return transformed_image

    def apply_corruption_on_folder(
        self,
        source_path: str,
        dest_path: str,
        corruption_strength: Union[float, int],
        show_progress: bool = True,
    ) -> str:
        """
        Apply a corruption to all images in a folder.

        :param source_path: Path to the folder containing the images (dataset format).
        :param dest_path: Path to the folder where the transformed images will be saved.
        :param corruption_strength: The strength of the corruption to apply. Range depends on the corruption type.
        :param show_progress: Whether to show a progress bar or not.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path '{source_path}' does not exist")
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for folder_name in tqdm(
            os.listdir(source_path),
            disable=not show_progress,
            desc="Applying Corruption",
        ):
            class_folder_path = os.path.join(source_path, folder_name)
            # loop through images in each class folder
            for image_name in os.listdir(class_folder_path):
                image = cv2.imread(os.path.join(class_folder_path, image_name))
                dest_folder = os.path.join(dest_path, folder_name)
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)

                transformed_image = self._apply_corruption_on_image(
                    image, corruption_strength=corruption_strength
                )
                # Save the transformed image to the destination folder
                cv2.imwrite(os.path.join(dest_folder, image_name), transformed_image)
        return dest_path

    def generate_ood_dataset_by_corruption(
        self,
        geti_deployment: Deployment,
        source_path: str,
        dest_path: str = None,
        desired_accuracy: float = 50,
        desired_accuracy_tol=3.0,
        show_progress: bool = True,
    ) -> str:
        """
        Generate a dataset of corrupted images from a source dataset of clean images.
        The corruption is applied until the classification accuracy (tp/tp+fp) on the
        generated dataset reached the desired accuracy.

        :geti_deployment: The trained geti deployment (model) to use for classification.
        :source_path: The path to the source dataset of clean images. The dataset is
            required to have the following structure:

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
            f"Generating OOD dataset by applying {self.corruption_type} corruption on {source_path}"
        )
        dataset_folder_name = os.path.basename(source_path)

        if dest_path is None:
            dest_path = os.path.join(
                os.path.dirname(source_path),
                f"{dataset_folder_name}_{self.corruption_type}_{desired_accuracy:.0f}",
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
        corruption_strength = self._corruption_strength_range[0]
        while abs(accuracy - desired_accuracy) > desired_accuracy_tol:
            corruption_strength = self._update_corruption_strength(
                desired_accuracy=desired_accuracy,
                current_accuracy=accuracy,
                current_strength=corruption_strength,
            )
            self.apply_corruption_on_folder(
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

    def _update_corruption_strength(
        self,
        desired_accuracy: float,
        current_accuracy: float,
        current_strength: float,
    ) -> float:
        """
        Update the corruption strength based on the current accuracy and the desired accuracy.
        """
        # The weight is used to control the speed of the corruption strength update.
        # If the current accuracy is close to the desired accuracy, a smaller weight is used to prevent overshoot.
        weight = 1.0 if abs(int((current_accuracy - desired_accuracy))) > 10 else 0.7

        limit = self._corruption_strength_range

        accuracy_diff = ((current_accuracy - desired_accuracy) / 100) * (
            limit[1] - limit[0]
        )
        updated_parameter = current_strength + accuracy_diff * weight

        updated_parameter = max(limit[0], min(limit[1], updated_parameter))
        return updated_parameter


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
        prediction = deployment.infer(numpy_rgb)
        pred_label = prediction.annotations[0].labels[0].name
        if pred_label == label:
            correct_classifications += 1
    accuracy = correct_classifications / len(id_images_dict)
    return accuracy * 100


def get_image_paths(src_dir, images_dict=None, label=None):
    """
    Recursively retrieve the paths of all images in a directory.

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


def extract_features_from_imageclient(
    deployment: Deployment,
    image_client: ImageClient,
    geti_session: GetiSession,
    n_images: int = -1,
    normalise_feats: bool = True,
):
    """
    Extract feature embeddings from a Geti deployment model for a given number of
    images in a geti image_client

    :param deployment: The trained Geti deployment (model) to use for feature extraction.
    :param image_client: The Geti ImageClient object containing the images to extract features from.
    :param geti_session: The GetiSession object.
    :param n_images: The number of images to extract features from.
        If -1, all images in the image_client will be used.
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
    Return the number of rows and columns for a grid arrangement of n items in a grid.
    This function returns the grid arrangement with the closest number of rows and
    columns to the square root of n.
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
