import copy
import time
from typing import List, Sequence, Tuple, Dict, Any, Optional, Union
import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories, \
    Bbox, Polygon
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from sc_api_tools.rest_converters import AnnotationRESTConverter

from .base_annotation_reader import AnnotationReader
from sc_api_tools.data_models import TaskType
from sc_api_tools.data_models import Annotation as SCAnnotation
from sc_api_tools.data_models.enums.task_type import GLOBAL_TASK_TYPES
from sc_api_tools.utils import get_dict_key_from_value, generate_segmentation_labels


class DatumAnnotationReader(AnnotationReader):
    """
    Class to read annotations using datumaro
    """

    _SUPPORTED_TASK_TYPES = [
        TaskType.DETECTION,
        TaskType.SEGMENTATION,
        TaskType.CLASSIFICATION,
        TaskType.ANOMALY_CLASSIFICATION
    ]

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
        self._override_label_map: Optional[Dict[int, str]] = None
        self._applied_filters: List[Dict[str, Union[List[str], str]]] = []

    def prepare_and_set_dataset(self, task_type: Union[TaskType, str]):
        if not isinstance(task_type, TaskType):
            task_type = TaskType(task_type)
        if task_type != self.task_type:
            print(f"Task type changed to {task_type} for dataset")
            if task_type not in self._SUPPORTED_TASK_TYPES:
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

    def convert_labels_to_segmentation_names(self) -> None:
        """
        This method converts the label names in a dataset to '*_shape`, where `*` is
        the original label name. It can be used to generate unique label names for the
        segmentation task in a detection_to_segmentation project
        """
        segmentation_label_map: Dict[int, str] = {}
        label_names = list(self.datum_label_map.values())
        segmentation_label_names = generate_segmentation_labels(label_names)
        for datum_index, label_name  in self.datum_label_map.items():
            label_index = label_names.index(label_name)
            segmentation_label_map.update(
                {datum_index: segmentation_label_names[label_index]}
            )
        self.override_label_map(segmentation_label_map)

    def get_all_label_names(self) -> List[str]:
        """
        Retrieves the list of labels names from a datumaro dataset
        """
        return list(set(self.datum_label_map.values()))

    @property
    def datum_label_map(self) -> Dict[int, str]:
        """
        :return: Dictionary mapping the datumaro label id to the label name
        """
        if self._override_label_map is None:
            return self.dataset.label_mapping
        else:
            return self._override_label_map

    def override_label_map(self, new_label_map: Dict[int, str]):
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

    def get_data(
            self,
            filename: str,
            label_name_to_id_mapping: dict,
            preserve_shape_for_global_labels: bool = False
    ) -> List[SCAnnotation]:
        ds_item = self.dataset.get_item_by_id(filename)
        image_size = ds_item.image.size
        annotation_list: List[SCAnnotation] = []
        labels = []
        for annotation in ds_item.annotations:
            try:
                label_name = self.datum_label_map[annotation.label]
            except KeyError:
                # Label is not in the SC project labels, move on to next annotation
                # for this dataset item.
                continue

            label_id = label_name_to_id_mapping.get(label_name)
            label = {'id': label_id, 'probability': 1.0}
            if (
                    self.task_type not in GLOBAL_TASK_TYPES
                    or preserve_shape_for_global_labels
            ):
                if isinstance(annotation, Bbox):
                    x1, y1 = annotation.points[0] / image_size[1], \
                             annotation.points[1] / image_size[0]
                    x2, y2 = annotation.points[2] / image_size[1], \
                             annotation.points[3] / image_size[0]
                    shape = {'type': 'RECTANGLE',
                             'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}
                elif isinstance(annotation, Polygon):
                    points = [
                        {'x': x/image_size[1], 'y': y/image_size[0]} for x, y
                        in zip(*[iter(annotation.points)] * 2)
                    ]
                    shape = {'type': "POLYGON", 'points': points}
                else:
                    print(
                        f"WARNING: Unsupported annotation type found: "
                        f"{type(annotation)}. Skipping..."
                    )
                    continue
                sc_annotation = AnnotationRESTConverter.annotation_from_dict(
                    {"labels": [label], "shape": shape}
                )
                annotation_list.append(sc_annotation)
            else:
                labels.append(label)

        if (
                not preserve_shape_for_global_labels
                and self.task_type in GLOBAL_TASK_TYPES
        ):
            shape = {
                "type": "RECTANGLE",
                "x": 0.0,
                "y": 0.0,
                "width": 1.0,
                "height": 1.0
            }
            sc_annotation = AnnotationRESTConverter.annotation_from_dict(
                {"labels": labels, "shape": shape}
            )
            annotation_list.append(sc_annotation)
        return annotation_list

    @property
    def applied_filters(self) -> List[Dict[str, Union[List[str], str]]]:
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

    def group_labels(self, labels_to_group: List[str], group_name: str) -> None:
        """
        Group multiple labels into one. Grouping converts the list of labels into one
        single label named `group_name`.

        This method does not return anything, but instead overrides the label map for
        the datamaro dataset to account for the grouping.

        :param labels_to_group: List of labels names that should be grouped together
        :param group_name: Name of the resulting label
        :return:
        """
        label_keys = [
            get_dict_key_from_value(self.datum_label_map, label)
            for label in labels_to_group
        ]
        new_label_map = copy.deepcopy(self.datum_label_map)
        for key in label_keys:
            new_label_map[key] = group_name
        self.override_label_map(new_label_map=new_label_map)


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
