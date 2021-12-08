from abc import abstractmethod
from typing import List


class AnnotationReader:
    """
    Base class for annotation reading, to handle loading and converting annotations
    to Sonoma Creek format
    """

    def __init__(
            self,
            base_data_folder: str,
            annotation_format: str = ".json",
            task_type: str = "detection"
    ):
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

    @abstractmethod
    def prepare_and_set_dataset(self, task_type: str):
        """
        Prepares a dataset for uploading annotations for a certain task_type
        :return:
        """
        raise NotImplementedError