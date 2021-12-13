import copy
import time
from typing import List, Sequence, Tuple, Dict, Any, Optional, Union
import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories, \
    Bbox, Polygon
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem

from .base_annotation_reader import AnnotationReader
from sc_api_tools.data_models import TaskType
from sc_api_tools.utils import get_dict_key_from_value, grouped


class DatumAnnotationReader(AnnotationReader):
    """
    Class to read annotations using datumaro
    """

    def __init__(
            self,
            base_data_folder: str,
            annotation_format: str,
            task_type: Union[TaskType, str] = TaskType.DETECTION
    ):
        super().__init__(
            base_data_folder=base_data_folder,
            annotation_format=annotation_format,
            task_type=task_type
        )
        self.dataset = DatumaroDataset(
            dataset_format=annotation_format, dataset_path=base_data_folder
        )
        self._override_label_map: Optional[Dict[str, int]] = None
        self._applied_filters: Optional[List[Dict[str, Union[List[str], str]]]] = []

    def prepare_and_set_dataset(self, task_type: Union[TaskType, str]):
        if not isinstance(task_type, TaskType):
            task_type = TaskType(task_type)
        if task_type != self.task_type:
            print(f"Task type changed to {task_type} for dataset")
            if task_type not in [TaskType.DETECTION, TaskType.SEGMENTATION]:
                raise ValueError(f"Unsupported task type {task_type}")
            new_dataset = DatumaroDataset(
                dataset_format=self.annotation_format, dataset_path=self.base_folder
            )
            self.task_type = task_type
            self.dataset = new_dataset
            for filter_parameters in self.applied_filters:
                self.filter_dataset(**filter_parameters)

        dataset = self.dataset.prepare_dataset(task_type=task_type)
        self.dataset.set_dataset(dataset)
        print(f"Dataset is prepared for {task_type} task.")

    def get_all_label_names(self) -> List[str]:
        """
        Retrieves the list of labels and the mapping of label names to integers from
        a datumaro dataset
        """
        return self.dataset.label_names

    @property
    def datum_label_map(self) -> Dict[str, int]:
        """
        :return: Dictionary mapping the label name to the datumaro label id
        """
        if self._override_label_map is None:
            return self.dataset.label_mapping
        else:
            return self._override_label_map

    def override_label_map(self, new_label_map: Dict[str, int]):
        """
        Overrides the label map defined in the datumaro dataset

        :return:
        """
        self._override_label_map = new_label_map

    def reset_label_map(self):
        """
        Resets the label map back to the original one from the datumaro dataset

        :return:
        """
        self._override_label_map = None

    def get_all_image_names(self) -> List[str]:
        """
        Returns a list of image names in the dataset

        :return:
        """
        return self.dataset.image_names

    def get_data(self, filename: str, label_name_to_id_mapping: dict):
        ds_item = self.dataset.get_item_by_id(filename)
        image_size = ds_item.image.size
        annotation_list: List[Dict[str, Any]] = []
        for annotation in ds_item.annotations:
            label_name = get_dict_key_from_value(
                self.datum_label_map, annotation.label
            )
            if label_name is None:
                # Label is not in the SC project labels, move on to next annotation
                # for this dataset item.
                continue

            label_id = label_name_to_id_mapping.get(label_name)

            if isinstance(annotation, Bbox):
                x1, y1 = annotation.points[0] / image_size[1], annotation.points[1] / \
                         image_size[0]
                x2, y2 = annotation.points[2] / image_size[1], annotation.points[3] / \
                         image_size[0]
                shape = {'type': 'RECTANGLE',
                         'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}

            elif isinstance(annotation, Polygon):
                points = [
                    {'x': x/image_size[1], 'y': y/image_size[0]} for x, y
                    in grouped(annotation.points, 2)
                ]
                shape = {'type': "POLYGON", 'points': points}
            else:
                print(
                    f"WARNING: Unsupported annotation type found: {type(annotation)}. "
                    f"Skipping..."
                )
                continue

            label = {'id': label_id, 'probability': 1.0}
            annotation_list.append({"labels": [label], "shape": shape})
        return annotation_list

    @property
    def applied_filters(self) -> Optional[List[Dict[str, Union[List[str], str]]]]:
        return copy.deepcopy(self._applied_filters)

    def filter_dataset(self, labels: Sequence[str], criterion='OR') -> None:
        """
        Retain only those items with annotations in the list of labels passed.

        :param: labels     List of labels to filter on
        :param: criterion  Filter criterion, currently "OR" or "AND" are implemented
        """
        self.dataset.filter_items_by_labels(
            labels=labels, criterion=criterion
        )
        self._applied_filters.append({"labels": labels, "criterion": criterion})


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
    def label_mapping(self) -> Dict[str, int]:
        """
        Returns the mapping of label name to label index
        :return:
        """
        return self.categories._indices

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

        :param: labels     List of labels to filter on
        :param: criterion  Filter criterion, currently "OR" or "AND" are implemented
        """
        label_map = self.label_mapping
        # Sanity check for filtering
        for label in labels:
            if label not in list(label_map.keys()):
                raise ValueError(
                    f'Cannot filter on label {label} because this is not in the '
                    f'dataset.'
                )

        if labels:
            def select_function(dataset_item, labels):
                # Filter function to apply to each item in the dataset
                item_labels = [
                    get_dict_key_from_value(label_map, x.label) for x
                    in dataset_item.annotations
                ]
                matches = []
                for label in labels:
                    if label in item_labels:
                        if criterion == 'OR':
                            return True
                        elif criterion == 'AND':
                            matches.append(True)
                        else:
                            raise ValueError(
                                'Invalid filter criterion, please select "OR" or "AND".'
                            )
                    else:
                        matches.append(False)
                return np.all(matches)

            # Messy way to manually keep track of labels and indices, must be a
            # better way in Datumaro but haven't found it yet
            label_categories = LabelCategories.from_iterable(labels)
            new_labelmap = {}
            for label in labels:
                new_labelmap[label] = label_map[label]
            label_categories._indices = new_labelmap
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
