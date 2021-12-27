import time
from typing import List, Dict, Tuple, Sequence, Optional

import numpy as np
from datumaro.components.annotation import LabelCategories, AnnotationType
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem

from sc_api_tools.data_models import TaskType
from sc_api_tools.data_models.enums.task_type import GLOBAL_TASK_TYPES
from sc_api_tools.utils import get_dict_key_from_value


class DatumaroDataset(object):
    def __init__(self, dataset_format: str, dataset_path: str):
        """
        Wrapper for interacting with the datumaro dataset, contains some example
        functions for dataset operations
        that can be carried out prior to importing the dataset into NOUS
        """
        self.dataset_format = dataset_format
        self.dataset_path = dataset_path
        self.dataset, self.environment = self.create_datumaro_dataset()
        self._subset_names = self.dataset.subsets().keys()

    def prepare_dataset(self, task_type: TaskType) -> Dataset:
        """
        Prepares the dataset for uploading to Sonoma Creek

        :param task_type: TaskType to prepare the dataset for
        """
        if task_type == TaskType.DETECTION:
            new_dataset = self.dataset.transform(
                self.dataset.env.transforms.get('shapes_to_boxes')
            )
            print("Annotations have been converted to boxes")
        elif task_type == TaskType.SEGMENTATION:
            new_dataset = self.dataset.transform(
                self.dataset.env.transforms.get('masks_to_polygons')
            )
            print("Annotations have been converted to polygons")
        elif task_type in GLOBAL_TASK_TYPES:
            new_dataset = self.dataset.transform(
                self.dataset.env.transforms.get('shapes_to_boxes')
            )
            print(f"{str(task_type).capitalize()} dataset prepared.")
        else:
            raise ValueError(f"Unsupported task type {task_type}")
        return new_dataset

    def set_dataset(self, dataset: Dataset):
        """
        Sets the dataset for this DatumaroDataset instance

        :param dataset:
        :return:
        """
        self.dataset = dataset
        self.environment = dataset.env
        self._subset_names = self.dataset.subsets().keys()

    @property
    def categories(self) -> LabelCategories:
        categories: LabelCategories = self.dataset.categories()[AnnotationType.label]
        return categories

    @property
    def label_names(self) -> List[str]:
        """
        Returns the labels
        :return:
        """
        return [item.name for item in self.categories]

    @property
    def label_mapping(self) -> Dict[int, str]:
        """
        Returns the mapping of label name to label index
        :return:
        """
        return {value: key for key, value in self.categories._indices.items()}

    @property
    def image_names(self) -> List[str]:
        """
        Return the list of media names included in the dataset.
        """
        return [item.id for item in self.dataset]

    def create_datumaro_dataset(self) -> Tuple[Dataset, Environment]:
        """
        Initialize a datumaro dataset.
        """
        t_start = time.time()
        dataset = Dataset.import_from(
            path=self.dataset_path, format=self.dataset_format
        )
        print(
            f'Datumaro dataset consisting of {len(dataset)} items in '
            f'{self.dataset_format} format was loaded from {self.dataset_path}'
        )
        print(f'Datumaro dataset was created in {time.time() - t_start:.1f} seconds')
        return dataset, dataset.env

    def remove_unannotated_items(self):
        """
        Keep only annotated images
        """
        self.dataset = self.dataset.select(lambda item: len(item.annotations) != 0)

    def filter_items_by_labels(self, labels: Sequence[str], criterion='OR') -> None:
        """
        Retain only those items with annotations in the list of labels passed.

        :param labels: List of labels to filter on
        :param criterion: Filter criterion, currently "OR", "NOT", "AND" and "XOR" are
            implemented
        """
        label_map = self.label_mapping
        # Sanity check for filtering
        for label in labels:
            if label not in list(label_map.values()):
                raise ValueError(
                    f'Cannot filter on label {label} because this is not in the '
                    f'dataset.'
                )

        if labels:
            def select_function(dataset_item: DatasetItem, labels: List[str]):
                # Filter function to apply to each item in the dataset
                item_labels = [
                    label_map[x.label] for x
                    in dataset_item.annotations
                ]
                matches = []
                for label in labels:
                    if label in item_labels:
                        if criterion == 'OR':
                            return True
                        elif criterion in ['AND', 'NOT', 'XOR']:
                            matches.append(True)
                        else:
                            raise ValueError(
                                'Invalid filter criterion, please select "OR", "NOT", '
                                '"XOR", or "AND".'
                            )
                    else:
                        matches.append(False)
                if criterion == 'AND':
                    return all(matches)
                elif criterion == 'NOT':
                    return not any(matches)
                elif criterion == 'XOR':
                    return np.sum(matches) == 1

            # Messy way to manually keep track of labels and indices, must be a
            # better way in Datumaro but haven't found it yet
            label_categories = LabelCategories.from_iterable(labels)
            new_labelmap = {}
            for label in labels:
                label_key = get_dict_key_from_value(label_map, label)
                new_labelmap[label_key] = label
            label_categories._indices = {v: k for k, v in new_labelmap.items()}
            new_categories = {AnnotationType.label: label_categories}
            # Filter and create a new dataset to update the dataset categories
            self.dataset = Dataset.from_iterable(
                self.dataset.select(lambda item: select_function(item, labels)),
                categories=new_categories,
                env=self.dataset.env
            )
            print(
                f'After filtering, dataset with labels {labels} contains '
                f'{len(self.dataset)} items.'
            )

    def get_item_by_id(self, datum_id: str) -> DatasetItem:
        ds_item: Optional[DatasetItem] = None
        for subset_name in self._subset_names:
            ds_item = self.dataset.get(id=datum_id, subset=subset_name)
        if ds_item is None:
            raise ValueError(
                f"Dataset item with id {datum_id} was not found in the dataset!"
            )
        return ds_item
