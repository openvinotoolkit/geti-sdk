import os
from typing import List, Optional, Sequence, Set, Union

from geti_sdk.annotation_readers import AnnotationReader
from geti_sdk.data_models import Annotation, TaskType


class DirectoryTreeAnnotationReader(AnnotationReader):
    """
    AnnotationReader for loading classification annotations from a dataset organized
    in a directory tree. This annotation reader expects images to be put in folders,
    where the name of each image folder corresponds to the label that should be
    assigned to all images inside it.

    :param base_data_folder: Root of the directory tree that contains the dataset
    :param subset_folder_names: Optional list of subfolders of the base_data_folder
        that should not be considered as labels, but should be used to acquire the
        data. For example ['train', 'validation', 'test'] for a dataset that is split
        into three subsets.
    :param task_type: TaskType for the task in the Intel® Geti™ platform to which the
        annotations should be uploaded
    """

    def __init__(
        self,
        base_data_folder: str,
        subset_folder_names: Optional[Sequence[str]] = None,
        task_type: Union[TaskType, str] = TaskType.CLASSIFICATION,
    ):
        if not task_type.is_global:
            raise ValueError(
                "The DirectoryTreeAnnotationReader only supports annotations for "
                "global task types."
            )
        super().__init__(
            base_data_folder=base_data_folder,
            annotation_format="directory",
            task_type=task_type,
        )

        has_root_level_subsets = True if subset_folder_names is not None else False
        if has_root_level_subsets:
            self.target_data_dirs = [
                os.path.join(base_data_folder, subset) for subset in subset_folder_names
            ]
        else:
            self.target_data_dirs = [base_data_folder]

    def get_data(
        self,
        filename: str,
        label_name_to_id_mapping: dict,
        preserve_shape_for_global_labels: bool = False,
    ) -> List[Annotation]:
        """
        Return the list of annotations for the media item with name `filename`

        :param filename: Name of the item to return the annotations for
        :param label_name_to_id_mapping: Dictionary mapping the name of a label to its
            unique database ID
        :param preserve_shape_for_global_labels: Unused parameter in this type of
            annotation reader
        :return: A list of Annotation objects for the media item
        """
        pass

    def get_all_label_names(self) -> List[str]:
        """
        Identify all label names contained in the dataset
        """
        label_names: Set[str] = set()
        for directory in self.target_data_dirs:
            for path, sub_directories, files in os.walk(directory):
                for sub_directory in sub_directories:
                    label_names.add(sub_directory)
        return list(label_names)

    def get_data_filenames(self) -> List[str]:
        """
        Return a list of annotated media files found in the dataset.

        :return: List of filenames (excluding extension) for all annotated files in
            the data folder
        """
        data_file_paths: List[str] = []
        for directory in self.target_data_dirs:
            for path, sub_directories, files in os.walk(directory):
                for name in files:
                    data_file_paths.append(
                        os.path.join(path, os.path.splitext(name)[0])
                    )
        return data_file_paths
