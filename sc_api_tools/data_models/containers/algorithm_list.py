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

import copy
from collections import UserList
from typing import Any, Dict, List, Optional, Sequence

from sc_api_tools.data_models.algorithms import Algorithm
from sc_api_tools.data_models.enums import TaskType

DEFAULT_ALGORITHMS = {
    "classification": "Custom_Image_Classification_EfficinetNet-B0",
    "detection": "Custom_Object_Detection_Gen3_ATSS",
    "segmentation": "Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR",
    "anomaly_classification": "ote_anomaly_classification_padim",
    "anomaly_detection": "ote_anomaly_classification_padim",
    "anomaly_segmentation": "ote_anomaly_segmentation_padim",
    "rotated_detection": "Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_ResNet50",
    "instance_segmentation": "Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50",
}


class AlgorithmList(UserList):
    """
    A list containing SC supported algorithms.
    """

    def __init__(self, data: Optional[Sequence[Algorithm]] = None):
        self.data: List[Algorithm] = []
        if data is not None:
            super().__init__(list(data))

    @staticmethod
    def from_rest(rest_input: Dict[str, Any]) -> "AlgorithmList":
        """
        Create an AlgorithmList from the response of the /supported_algorithms REST
        endpoint in SC.

        :param rest_input: Dictionary retrieved from the /supported_algorithms REST
            endpoint
        :return: AlgorithmList holding the information related to the supported
            algorithms in SC
        """
        algorithm_list = AlgorithmList([])
        try:
            algo_rest = rest_input["items"]
        except KeyError:
            algo_rest = rest_input["supported_algorithms"]
        algo_rest_list = copy.deepcopy(algo_rest)
        for algorithm_dict in algo_rest_list:
            algorithm_list.append(Algorithm(**algorithm_dict))
        return algorithm_list

    def get_by_model_template(self, model_template_id: str) -> Algorithm:
        """
        Retrieve an algorithm from the list by its model_template_id.

        :param model_template_id: Name of the model template to get the Algorithm
            information for
        :return: Algorithm holding the algorithm details
        """
        for algo in self.data:
            if algo.model_template_id == model_template_id:
                return algo
        raise ValueError(
            f"Algorithm for model template {model_template_id} was not found in the "
            f"list of supported algorithms."
        )

    def get_by_task_type(self, task_type: TaskType) -> "AlgorithmList":
        """
        Return a list of supported algorithms for a particular task type.

        :param task_type: TaskType to get the supported algorithms for
        :return: List of supported algorithms for the task type
        """
        return AlgorithmList(
            [algo for algo in self.data if algo.task_type == task_type]
        )

    @property
    def summary(self) -> str:
        """
        Return a string that gives a very brief summary of the algorithm list.

        :return: String holding a brief summary of the list of algorithms
        """
        summary_str = "Algorithms:\n"
        for algorithm in self.data:
            summary_str += (
                f"  Name: {algorithm.algorithm_name}\n"
                f"    Task type: {algorithm.task_type}\n"
                f"    Model size: {algorithm.model_size}\n"
                f"    Gigaflops: {algorithm.gigaflops}\n\n"
            )
        return summary_str

    def get_by_name(self, name: str) -> Algorithm:
        """
        Retrieve an algorithm from the list by its algorithm_name.

        :param name: Name of the Algorithm to get
        :return: Algorithm holding the algorithm details
        """
        for algo in self.data:
            if algo.algorithm_name == name:
                return algo
        raise ValueError(
            f"Algorithm named {name} was not found in the "
            f"list of supported algorithms."
        )

    def get_default_for_task_type(self, task_type: TaskType) -> Algorithm:
        """
        Return the default algorithm for a given task type. If there is no algorithm
        for the task type in the AlgorithmList, this method will raise a ValueError.

        :param task_type: TaskType of the task to get the default algorithm for
        :raises: ValueError if there are no available algorithms for the specified
            task_type in the AlgorithmList
        :return: Default algorithm for the task
        """
        return self.get_by_model_template(DEFAULT_ALGORITHMS[str(task_type)])
