import logging
import os
import warnings
from glob import glob
from typing import Dict, List, Optional, Sequence, Set, Union

import cv2

from geti_sdk.annotation_readers import AnnotationReader
from geti_sdk.data_models import Annotation, ScoredLabel, TaskType
from geti_sdk.data_models.shapes import Rectangle


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
        # Label map is a dictionary mapping the root labels to new label names. It
        # can be used to filter or group the labels in the dataset
        self._label_map: Dict[str, str] = {}
        self._original_labels = self.get_all_label_names()

    @property
    def label_map(self) -> Dict[str, str]:
        """
        Return the label map for the dataset, mapping the root label names (keys) to
        potential new label names (values). It is used to filter or group the dataset.

        If no filters or grouping has been applied, it returns a dictionary with key,
        value pairs that have identical keys and values, i.e. {"dog": "dog"}
        """
        if not self._label_map:
            return {label: label for label in self._original_labels}
        else:
            return self._label_map

    def reset_filters_and_grouping(self):
        """
        Reset the applied filters and grouping, to recover the original dataset
        """
        self._label_map = {}

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
        matches = glob(
            os.path.join(self.base_folder, "**", f"{filename}.*"), recursive=True
        )
        label_matches = [os.path.basename(os.path.dirname(match)) for match in matches]
        annotations: List[Annotation] = []
        if len(label_matches) > 1:
            warnings.warn(
                f"Multiple matching labels found for image with "
                f"name {filename}: {label_matches}. Skipping this image..."
            )
        elif len(label_matches) == 0:
            logging.info(
                f"Image with name {filename} was not found in the dataset at path "
                f"{self.base_folder}. Skipping this image..."
            )
        else:
            img = cv2.imread(matches[0])
            width, height = img.shape[1, 0]
            if self.label_map:
                label_name = self.label_map[label_matches[0]]
            else:
                label_name = label_matches[0]
            label = ScoredLabel(
                name=label_name,
                probability=1.0,
                id=label_name_to_id_mapping[label_name],
            )
            annotations.append(
                Annotation(
                    labels=[label],
                    shape=Rectangle(x=0, y=0, width=width, height=height),
                )
            )
        return annotations

    def get_all_label_names(self) -> List[str]:
        """
        Identify all label names contained in the dataset
        """
        label_names: Set[str] = set()
        requires_filter = False
        if self._label_map:
            requires_filter = True
        for directory in self.target_data_dirs:
            for path, sub_directories, files in os.walk(directory):
                for sub_directory in sub_directories:
                    if requires_filter:
                        if sub_directory not in self.label_map.keys():
                            continue
                    label_names.add(sub_directory)
        return list(label_names)

    def get_data_filenames(self) -> List[str]:
        """
        Return a list of annotated media files found in the dataset.

        :return: List of filenames (excluding extension) for all annotated files in
            the data folder
        """
        data_file_paths: List[str] = []
        requires_filter = False
        if self._label_map:
            requires_filter = True
        for directory in self.target_data_dirs:
            for path, sub_directories, files in os.walk(directory):
                if requires_filter:
                    if (
                        os.path.basename(os.path.dirname(path))
                        not in self.label_map.keys()
                    ):
                        continue
                for name in files:
                    data_file_paths.append(
                        os.path.join(path, os.path.splitext(name)[0])
                    )
        return data_file_paths

    def filter_dataset(self, labels: Sequence[str]) -> None:
        """
        Retain only those items with annotations in the list of labels passed.

        :param labels: List of labels to filter on
        """
        self._label_map = {label: label for label in labels}

    def group_labels(self, labels_to_group: List[str], group_name: str) -> None:
        """
        Group multiple labels into one. Grouping converts the list of labels into one
        single label named `group_name`.

        This method does not return anything, but instead overrides the label map for
        the annotation reader to account for the grouping.

        :param labels_to_group: List of labels names that should be grouped together
        :param group_name: Name of the resulting label
        """
        for org_label_name, mapped_label in self.label_map.items():
            if org_label_name in labels_to_group:
                self._label_map.update({org_label_name: group_name})
