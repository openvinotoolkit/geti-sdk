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
import shutil
import tempfile
from typing import List

import cv2
import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from geti_sdk.data_models import Project
from tests.helpers.constants import PROJECT_PREFIX
from tests.helpers.project_service import ProjectService


class TestImageClient:
    @staticmethod
    def ensure_test_project(
        project_service: ProjectService, labels: List[str]
    ) -> Project:
        return project_service.get_or_create_project(
            project_name=f"{PROJECT_PREFIX}_image_client",
            project_type="detection",
            labels=[labels],
        )

    @pytest.mark.vcr()
    def test_upload_and_delete_image(
        self,
        fxt_project_service: ProjectService,
        fxt_default_labels: List[str],
        fxt_image_path: str,
    ):
        """
        Verifies that uploading an image works as expected

        Steps:
        1. Create detection project
        2. Upload image from file
        3. Load image with OpenCV directly
        4. Assert that the returned image details (filename, media dimensions) are
            correct
        5. Fetch numpy data for image from server
        6. Check that the two image arrays are approximately equal: There may be
            small difference due to image compression. We test for equality by
            asserting that the RMS error between the two numpy image arrays is less
            than 1
        7. Assert that deleting the image works by checking that the number of images
            in the project reduces by one upon deletion of the image
        """
        self.ensure_test_project(
            project_service=fxt_project_service, labels=fxt_default_labels
        )
        image_client = fxt_project_service.image_client
        image = image_client.upload_image(fxt_image_path)
        image_numpy = cv2.imread(fxt_image_path)

        assert image.name == os.path.splitext(os.path.basename(fxt_image_path))[0]
        assert image.media_information.width == image_numpy.shape[1]
        assert image.media_information.height == image_numpy.shape[0]

        fetched_numpy = image.get_data(session=fxt_project_service.session)
        assert fetched_numpy.shape == image_numpy.shape
        difference = fetched_numpy.astype(int) - image_numpy.astype(int)

        assert np.sqrt(np.mean(np.square(difference))) < 1

        n_images = len(image_client.get_all_images())
        result = image_client.delete_images([image])
        assert result
        assert len(image_client.get_all_images()) == n_images - 1

    @pytest.mark.vcr()
    def test_upload_image_folder_and_download(
        self,
        fxt_project_service: ProjectService,
        fxt_default_labels: List[str],
        fxt_image_folder: str,
        request: FixtureRequest,
    ):
        """
        Verifies that uploading a folder of images works as expected

        Steps:
        1. Get project and image client
        2. Get list of all images in project
        3. Upload folder containing several images
        4. Get list of all images in project again
        5. Assert that list of images has increased by the number of images in the
            folder
        6. Download the images to a temporary directory
        7. Assert that all images were downloaded
        """
        self.ensure_test_project(
            project_service=fxt_project_service, labels=fxt_default_labels
        )
        image_client = fxt_project_service.image_client

        old_images = image_client.get_all_images()
        n_images = len(old_images)

        # Upload folder
        images = image_client.upload_folder(fxt_image_folder)
        assert len(images) == len(os.listdir(fxt_image_folder))
        assert len(image_client.get_all_images()) == n_images + len(images)

        target_dir = tempfile.mkdtemp()
        request.addfinalizer(lambda: shutil.rmtree(target_dir))

        # Download all images
        image_client.download_all(target_dir)

        # Assert that all images are downloaded
        downloaded_filenames = os.listdir(os.path.join(target_dir, "images"))
        assert len(downloaded_filenames) == n_images + len(images)
        for image in images + old_images:
            assert image.name + ".jpg" in downloaded_filenames
