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

import glob
import json
import os
from typing import List, Optional, Union

from sc_api_tools.rest_converters import AnnotationRESTConverter

from .base_annotation_reader import AnnotationReader
from sc_api_tools.data_models import TaskType, Annotation


class SCAnnotationReader(AnnotationReader):
    def __init__(
            self,
            base_data_folder: str,
            annotation_format: str = ".json",
            task_type: Optional[Union[TaskType, str]] = None,
            label_names_to_include: Optional[List[str]] = None
    ):
        if annotation_format != '.json':
            raise ValueError(
                f"Annotation format {annotation_format} is currently not"
                f" supported by the SCAnnotationReader"
            )
        super().__init__(
            base_data_folder=base_data_folder,
            annotation_format=annotation_format,
            task_type=task_type
        )
        self._label_names_to_include = label_names_to_include

    def _get_label_names(self, all_labels: List[str]) -> List[str]:
        """
        Returns the labels for the task type the annotation reader is currently set to.

        :param all_labels: List of all label names in the project
        """
        if self._label_names_to_include is None:
            print("No label mapping defined, including all labels")
            labels = all_labels
        else:
            labels = [
                label for label in all_labels
                if label in self._label_names_to_include
            ]
        return labels

    def get_data(
            self,
            filename: str,
            label_name_to_id_mapping: dict,
            preserve_shape_for_global_labels: bool = False
    ) -> List[Annotation]:
        filepath = glob.glob(
            os.path.join(
                self.base_folder, f"{filename}{self.annotation_format}")
        )
        if len(filepath) > 1:
            print(
                f"Multiple matching annotation files found for image with "
                f"name {filename}. Skipping this image..."
            )
            return []
        elif len(filepath) == 0:
            print(
                f"No matching annotation file found for image with name {filename}."
                f" Skipping this image..."
            )
            return []
        else:
            filepath = filepath[0]
        with open(filepath, 'r') as f:
            data = json.load(f)

        new_annotations = []
        for annotation in data["annotations"]:
            annotation_object = AnnotationRESTConverter.annotation_from_dict(annotation)
            for label in annotation_object.labels:
                label.id = label_name_to_id_mapping[label.name]
            for label_dict in annotation["labels"]:
                if self.task_type is not None:
                    if label_dict["name"] not in self._get_label_names(
                            list(label_name_to_id_mapping.keys())
                    ):
                        annotation_object.pop_label_by_name(
                            label_name=label_dict["name"]
                        )
            new_annotations.append(annotation_object)
        return new_annotations

    def get_all_label_names(self) -> List[str]:
        """
        Retrieve the unique label names for all annotations in the annotation folder

        :return: List of label names
        """
        print(f"Reading annotation files in folder {self.base_folder}...")
        unique_label_names = []
        annotation_files = glob.glob(
            os.path.join(self.base_folder, f"*{self.annotation_format}")
        )
        if len(annotation_files) == 0:
            raise ValueError(
                f"No valid annotation files were found in folder {self.base_folder}"
            )
        for annotation_file in annotation_files:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            annotations = data.get("annotations", None)
            if annotations is None:
                raise ValueError(
                    f"Annotation file '{annotation_file}' does not contain any "
                    f"annotations. Please make sure that this is a valid "
                    f"annotation file."
                )
            for annotation in annotations:
                labels = [label["name"] for label in annotation["labels"]]
                for label in labels:
                    if label not in unique_label_names:
                        unique_label_names.append(label)
        return unique_label_names
