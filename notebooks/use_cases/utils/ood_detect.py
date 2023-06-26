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

import matplotlib.pyplot as plt
import numpy as np
import cv2
from .image import get_image_paths, get_grid_arrangement


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
        raise ValueError(f"type_of_samples must be one of ['id', 'ood'], got {type_of_samples}")

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
        ax.set_title(f"Label: {label} \n (Sore : {scores[score_sort_inds[i]]:.2f})",
                     color='#0068b5',
                     fontsize=11, wrap=True)
        ax.axis('off')
    fig.suptitle(f"Top {n_images} misclassified {type_of_samples} images", fontsize=16)
    plt.tight_layout()
    plt.show()
