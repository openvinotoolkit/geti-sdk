from abc import abstractmethod
from typing import List, Union, Dict

from sc_api_tools.data_models import TaskType


class AnnotationReader:
    """
    Base class for annotation reading, to handle loading and converting annotations
    to Sonoma Creek format
    """

    def __init__(
            self,
            base_data_folder: str,
            annotation_format: str = ".json",
            task_type: Union[TaskType, str] = TaskType.DETECTION
    ):
        if not isinstance(task_type, TaskType):
            task_type = TaskType(task_type)
        self.base_folder = base_data_folder
        self.annotation_format = annotation_format
        self.task_type = task_type

    @abstractmethod
    def get_data(self, filename: str, label_name_to_id_mapping: dict):
        """
        Get annotation data for a certain filename
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_label_names(self) -> List[str]:
        """
        Returns a list of unique label names that were found in the annotation data
        folder belonging to this AnnotationReader instance

        :return:
        """
        raise NotImplementedError

    def prepare_and_set_dataset(self, task_type: Union[TaskType, str]):
        """
        Prepares a dataset for uploading annotations for a certain task_type

        :return:
        """
        if not isinstance(task_type, TaskType):
            task_type = TaskType(task_type)
        if task_type in [TaskType.DETECTION, TaskType.SEGMENTATION]:
            self.task_type = task_type
        else:
            raise ValueError(f"Unsupported task_type {task_type}")

    @property
    def applied_filters(self) -> List[Dict[str, Union[List[str], str]]]:
        """
        Returns a list of dictionaries representing the filter settings that have
        been applied to the dataset, if any.

        Dictionaries in this list contain two keys:
        - 'labels'      -- List of label names which has been filtered on
        - 'criterion'   -- String representing the criterion that has been used in the
                           filtering. Can be 'OR', 'AND', 'XOR' or 'NOT'.

        :return: List of filter settings that have been applied to the dataset. Returns
            an empty list if no filters have been applied.
        """
        return []
