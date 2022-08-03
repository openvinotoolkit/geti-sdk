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

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem

from sc_api_tools.data_models import TaskType
from sc_api_tools.utils import get_dict_key_from_value


class DatumaroDataset(object):
    """
    Wrapper for interacting with the datumaro dataset, contains some example
    functions for dataset operations that can be carried out prior to importing the
    dataset into SonomaCreek
    """

    def __init__(self, dataset_format: str, dataset_path: str):
        """
        Initialize the datumaro dataset.

        :param dataset_format: string containing the format of the dataset, i.e.
            'voc', 'coco', 'imagenet', etc.
        :param dataset_path: Path to the dataset on local disk
        """
        self.dataset_format = dataset_format
        self.dataset_path = dataset_path
        self.dataset, self.environment = self.create_datumaro_dataset()
        self._subset_names = self.dataset.subsets().keys()

    def prepare_dataset(
        self, task_type: TaskType, previous_task_type: Optional[TaskType] = None
    ) -> Dataset:
        """
        Prepare the dataset for uploading to Sonoma Creek.

        :param task_type: TaskType to prepare the dataset for
        :param previous_task_type: Optional type of the (trainable) task preceding
            the current task in the pipeline. This is only used for global tasks
        """
        if task_type.is_detection:
            new_dataset = self.dataset.transform(
                self.dataset.env.transforms.get("shapes_to_boxes")
            )
            logging.info("Annotations have been converted to boxes")
        elif task_type.is_segmentation:
            new_dataset = self.dataset.transform(
                self.dataset.env.transforms.get("masks_to_polygons")
            )
            logging.info("Annotations have been converted to polygons")
        elif task_type.is_global:
            if previous_task_type is not None and previous_task_type.is_segmentation:
                new_dataset = self.dataset.transform(
                    self.dataset.env.transforms.get("masks_to_polygons")
                )
            else:
                new_dataset = self.dataset.transform(
                    self.dataset.env.transforms.get("shapes_to_boxes")
                )
            logging.info(f"{str(task_type).capitalize()} dataset prepared.")
        else:
            raise ValueError(f"Unsupported task type {task_type}")
        return new_dataset

    def set_dataset(self, dataset: Dataset):
        """
        Set the dataset for this DatumaroDataset instance.

        :param dataset:
        :return:
        """
        self.dataset = dataset
        self.environment = dataset.env
        self._subset_names = self.dataset.subsets().keys()

    @property
    def categories(self) -> LabelCategories:
        """
        Return the LabelCategories in the dataset.
        """
        categories: LabelCategories = self.dataset.categories()[AnnotationType.label]
        return categories

    @property
    def label_names(self) -> List[str]:
        """
        Return a list of all label names in the dataset.
        """
        return [item.name for item in self.categories]

    @property
    def label_mapping(self) -> Dict[int, str]:
        """
        Return the mapping of label index to label name.
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
        logging.info(
            f"Datumaro dataset consisting of {len(dataset)} items in "
            f"{self.dataset_format} format was loaded from {self.dataset_path}"
        )
        logging.info(f"Datumaro dataset was created in {time.time() - t_start:.1f} seconds")
        return dataset, dataset.env

    def remove_unannotated_items(self):
        """
        Keep only annotated images.
        """
        self.dataset = self.dataset.select(lambda item: len(item.annotations) != 0)

    def filter_items_by_labels(self, labels: Sequence[str], criterion="OR") -> None:
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
                    f"Cannot filter on label {label} because this is not in the "
                    f"dataset."
                )

        if labels:

            def select_function(dataset_item: DatasetItem, labels: List[str]):
                # Filter function to apply to each item in the dataset
                item_labels = [label_map[x.label] for x in dataset_item.annotations]
                matches = []
                for label in labels:
                    if label in item_labels:
                        if criterion == "OR":
                            return True
                        elif criterion in ["AND", "NOT", "XOR"]:
                            matches.append(True)
                        else:
                            raise ValueError(
                                'Invalid filter criterion, please select "OR", "NOT", '
                                '"XOR", or "AND".'
                            )
                    else:
                        matches.append(False)
                if criterion == "AND":
                    return all(matches)
                elif criterion == "NOT":
                    return not any(matches)
                elif criterion == "XOR":
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
                env=self.dataset.env,
            )
            logging.info(
                f"After filtering, dataset with labels {labels} contains "
                f"{len(self.dataset)} items."
            )

    def get_item_by_id(self, datum_id: str) -> DatasetItem:
        """
        Return the dataset item by its id.

        :param datum_id: Datumaro id of the item to retrieve
        :raises: ValueError if no item by that id was found.
        :return: DatasetItem with the given id
        """
        ds_item: Optional[DatasetItem] = None
        for subset_name in self._subset_names:
            ds_item = self.dataset.get(id=datum_id, subset=subset_name)
        if ds_item is None:
            raise ValueError(
                f"Dataset item with id {datum_id} was not found in the dataset!"
            )
        return ds_item
