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

import pandas as pd
from test_nightly_project import TestNightlyProject

from geti_sdk.benchmarking.benchmarker import Benchmarker
from geti_sdk.geti import Geti
from tests.helpers import project_service


class TestClassification(TestNightlyProject):
    """
    Class to test project creation, annotation upload, training, prediction, benchmarking and
    deployment for a classification project
    """

    PROJECT_TYPE = "classification"
    __test__ = True

    def test_benchmarking(
        self,
        fxt_project_service_no_vcr: project_service,
        fxt_geti_no_vcr: Geti,
        fxt_temp_directory: str,
        fxt_image_path: str,
        fxt_image_path_complex: str,
        fxt_artifact_directory: str,
    ):
        """
        Tests benchmarking for the project.
        """
        project = fxt_project_service_no_vcr.project
        algorithms_to_benchmark = ["EfficientNet-B0", "MobileNet-V3-large-1x"]
        images = [fxt_image_path, fxt_image_path_complex]
        precision_levels = ["FP32", "FP16", "INT8"]

        benchmarker = Benchmarker(
            geti=fxt_geti_no_vcr,
            project=project,
            algorithms=algorithms_to_benchmark,
            precision_levels=precision_levels,
            benchmark_images=images,
        )
        benchmarker.prepare_benchmark(working_directory=fxt_temp_directory)
        results = benchmarker.run_throughput_benchmark(
            working_directory=fxt_temp_directory,
            results_filename="results",
            target_device="CPU",
            frames=2,
            repeats=2,
        )
        pd.DataFrame(results)
