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
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from geti_sdk.data_models import Prediction
from geti_sdk.deployment.inference_hook_interfaces import PostInferenceAction
from geti_sdk.rest_converters import PredictionRESTConverter


class FileSystemDataCollection(PostInferenceAction):
    """
    Post inference action that will save an image to a specified folder on disk. The
    prediction output and trigger score that triggered the action are also saved.

    The data is saved in the `target_folder`, in which the action will create the
    following folder structure:

    <target_folder>
        |
        |- images
        |- predictions
        |- scores

    :param target_folder: Target folder on disk where the inferred images should be
        saved. If it does not exist yet, this action will create it.
    """

    def __init__(self, target_folder: str, file_name_prefix: str = "image"):
        self.image_path = os.path.join(target_folder, "images")
        self.predictions_path = os.path.join(target_folder, "predictions")
        self.scores_path = os.path.join(target_folder, "scores")

        for path in [self.image_path, self.predictions_path, self.scores_path]:
            os.makedirs(path, exist_ok=True)

        self.prefix = file_name_prefix

    def __call__(
        self, image: np.ndarray, prediction: Prediction, score: Optional[float] = None
    ):
        """
        Execute the action, save the given `image` to the predefined target folder.
        The `prediction` and `score` are also saved.

        :param image: Numpy array representing an image
        :param prediction: Prediction object which was generated for the image
        :param score: Optional score computed from a post inference trigger
        """
        image_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # upload_image uses cv2 to encode the numpy array as image, so it expects an
        # image in BGR format. However, `Deployment.infer` requires RGB format, so
        # we have to convert
        filename = self.prefix + "_" + datetime.now().__str__()
        cv2.imwrite(os.path.join(self.image_path, filename + ".png"), image_bgr)
        logging.info(
            f"FileSystemDataCollection inference action saved image data to folder "
            f"`{self.image_path}`"
        )
        prediction_filepath = os.path.join(self.predictions_path, filename + ".json")
        with open(prediction_filepath, "w") as file:
            prediction_dict = PredictionRESTConverter.to_dict(prediction)
            json.dump(file, prediction_dict)

        if score is not None:
            score_filepath = os.path.join(self.scores_path, filename + ".txt")
            with open(score_filepath, "w") as file:
                file.write(f"score={score:.4f}")

    def __repr__(self):
        """
        Return a string representation of the GetiDataCollection action object
        """
        return (
            f"PostInferenceAction `FileSystemDataCollection`"
            f"(target_folder={self.image_path})"
        )
