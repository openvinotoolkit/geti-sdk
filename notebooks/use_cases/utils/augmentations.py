import os
from typing import Union

import cv2
import imgaug.augmenters as iaa
import numpy as np
from tqdm import tqdm

from geti_sdk.data_models import Image

from .image import image_to_np


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

        self.corruption_strength_range = (1, 300)
        if self.corruption_type in ["cut_out", "fake_snow"]:
            self.corruption_strength_range = (0.001, 1)
        elif self.corruption_type == "poisson_noise":
            self.corruption_strength_range = (
                1,
                110,
            )  # Increase this if the desired accuracy is not reached
        elif self.corruption_type == "gaussian_blur":
            self.corruption_strength_range = (
                1,
                8,
            )  # Increase this if the desired accuracy is not reached
        elif self.corruption_type == "motion_blur":
            self.corruption_strength_range = (
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
        input_image = image_to_np(image)
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

    def update_corruption_strength(
        self,
        desired_accuracy: float,
        current_accuracy: float,
        current_strength: float,
    ) -> float:
        """
        Heuristic to update the corruption strength based on the current accuracy and the desired accuracy.
        """
        # The weight is used to control the speed of the corruption strength update.
        # If the current accuracy is close to the desired accuracy, a smaller weight is used to prevent overshoot.
        weight = 1.0 if abs(int((current_accuracy - desired_accuracy))) > 10 else 0.7

        limit = self.corruption_strength_range

        accuracy_diff = ((current_accuracy - desired_accuracy) / 100) * (
            limit[1] - limit[0]
        )
        updated_parameter = current_strength + accuracy_diff * weight

        updated_parameter = max(limit[0], min(limit[1], updated_parameter))
        return updated_parameter
