import glob
import json
import os
import re
from typing import Tuple, List, Dict, Any, Union, Optional

from sc_api_tools.data_models import TaskType
from sc_api_tools.rest_converters import AnnotationRESTConverter

from .base_annotation_reader import AnnotationReader


class NOUSAnnotationReader(AnnotationReader):
    """
    Class to read annotations for the NOUS dataset
    """
    def __init__(
            self,
            base_data_folder: str,
            annotation_format: str = ".json",
            task_type: Union[TaskType, str] = TaskType.SEGMENTATION
    ):
        super().__init__(
            base_data_folder=base_data_folder,
            annotation_format=annotation_format,
            task_type=task_type
        )

        self.label_filter: Optional[List[str]] = None

    def replace_empty_with_no_object(self, label):

        if self.task_type == TaskType.DETECTION:
            if 'Empty' in label and str(self.task_type) in str.lower(label):
                return 'No Object'

        if self.task_type == TaskType.CLASSIFICATION:
            if 'Empty' in label and str(self.task_type) in str.lower(label):
                return 'No Class'

        if self.task_type == TaskType.SEGMENTATION:
            if 'Empty' in label and str(self.task_type) in str.lower(label):
                return 'Empty'

        return label

    def get_all_label_names(self):
        '''
        Gets all the NOUS labels in a project and removes labels that contain 'Empty' and Task (i.e. 'detection')
        '''
        print(f"Reading annotation files in folder {self.base_folder}...")
        unique_label_names = []
        for annotation_file in os.listdir(self.base_folder):
            with open(os.path.join(self.base_folder, annotation_file), 'r') as f:
                data = json.load(f)
            for entry in data["data"]:
                labels = [label["name"] for label in entry["labels"]]
                for label in labels:
                    if 'Empty' in label and str(self.task_type) in str.lower(label):
                        continue
                    if label not in unique_label_names:
                        unique_label_names.append(label)
        return unique_label_names

    @staticmethod
    def _convert_filename(filename: str) -> Tuple[str, str, str]:
        """
        Splits a filename for an image exported from NOUS to match its corresponding
        annotation file name. The image filename is split into parts: It might
        contain a single UID or two UID's (in case of a video frame).

        The goal is to get rid of the uid that is appended to the end of the filename.

        :param filename: Image/videoframe filename to convert
        :return: Tuple containing:
            - The first part of the filename. Typically this should be the actual
                filename of the image, without the UID
            - The name_separator token that separates the uid from the filename. This
                should be '_'
            - The UID part of the filename.
        """
        name_separator = '_'
        uuid_pattern = re.compile("[0-9a-fA-F]{24}")
        uuid_matches = uuid_pattern.search(filename)
        uuid_start_index = uuid_matches.span()[0]
        filename_no_uid = filename[:uuid_start_index-1]
        return filename_no_uid, name_separator, uuid_matches.group()

    def _get_new_shape(self, x: float, y: float, radius: float) -> dict:
        """
        Converts a NOUS point shape into SC shape format, for the appropriate task type.

        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :param radius: radius of the point
        :return: dictionary containing the new SC shape data
        """
        if self.task_type == TaskType.SEGMENTATION:
            new_shape = {
                "type": "ELLIPSE",
                "width": 2 * radius,
                "height": 2 * radius,
                "x": x - radius,
                "y": y - radius
            }
        elif self.task_type == TaskType.DETECTION:
            new_shape = {
                "type": "RECTANGLE",
                "x": x - radius,
                "y": y - radius,
                "width": 2 * radius,
                "height": 2 * radius
            }
        else:
            raise ValueError(
                f"Unsupported task type set in annotation reader: {self.task_type}"
            )
        return new_shape

    def _get_new_rect(self, x: float, y: float, width: float, height: float) -> dict:
        """
        Converts a NOUS point shape into SC shape format, for the appropriate task type.

        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :param radius: radius of the point
        :return: dictionary containing the new SC shape data
        """

        if x+width > 1:
            width = 1-x

        if y+height > 1:
            height = 1-y

        if self.task_type == TaskType.DETECTION or \
            self.task_type == TaskType.CLASSIFICATION or \
            self.task_type == TaskType.SEGMENTATION:
            new_shape = {
                "type": "RECTANGLE",
                "x": x,
                "y": y,
                "width": width,
                "height": height
            }
        else:
            raise ValueError(
                f"Unsupported task type set in annotation reader: {self.task_type}"
            )
        return new_shape

    def _get_new_polygon(self, points: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Converts a NOUS polygon shape into SC shape format, for the appropriate task
        type.

        :param points: List of points making up the NOUS polygon
        :return: dictionary containing the new SC shape data
        """
        if self.task_type == TaskType.SEGMENTATION:
            shape = {"type": "POLYGON", "points": points}
        elif self.task_type == TaskType.DETECTION:
            shape = {"type": "RECTANGLE"}
            x_coordinates = [point["x"] for point in points]
            y_coordinates = [point["y"] for point in points]
            x_min, x_max = min(x_coordinates), max(x_coordinates)
            y_min, y_max = min(y_coordinates), max(y_coordinates)
            shape.update(
                {
                    "x": x_min,
                    "y": y_min,
                    "width": x_max - x_min,
                    "height": y_max - y_min
                }
            )
        else:
            raise ValueError(
                f"Unsupported task type set in annotation reader: {self.task_type}"
            )
        return shape

    def set_labels_filter(self, labels: Optional[List[str]] = None):
        """
        Sets the annotation reader to use only the labels passed in `labels`

        :param labels: Label names to filter on
        """
        self.label_filter = labels

    def get_data(
            self,
            filename: str,
            label_name_to_id_mapping: dict,
            preserve_shape_for_global_labels: bool = False,
            frame: int = -1
    ):
        media_filename, separator_token, uuid = self._convert_filename(
            filename
        )

        if frame != -1:
            annotation_filename = f'{media_filename}_frame_{frame}'
        else:
            annotation_filename = media_filename

        annotation_files = [
            self._convert_filename(filename)[0] for filename
            in os.listdir(self.base_folder)
        ]

        filepath = glob.glob(
            os.path.join(self.base_folder, f"{annotation_filename}*")
        )

        annotation_list = []
        if annotation_filename not in annotation_files or len(filepath) == 0:
            print(f"No annotation file found for image {filename}.")
        else:
            if len(filepath) != 1:
                # Try to match the annotation files against the media uuid
                for filename in filepath:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    if data.get("image_id", None) == uuid:
                        filepath = [filename]
                        break
                    if data.get("video_id", None) == uuid:
                        frame_idx = int(data.get("frame_id", -2))
                        if frame_idx == frame:
                            filepath = [filename]
                            break
                if len(filepath) == 0:
                    print(
                        f"No annotation file found for image or video frame "
                        f"{full_filename}."
                    )
                    return []
                elif len(filepath) != 1:
                    print(
                        f"Multiple matching annotation files found for image with "
                        f"name {full_filename}. Skipping this image..."
                    )
                    return []

            filepath = filepath[0]
            with open(filepath, 'r') as f:
                data = json.load(f)
            for entry in data["data"]:
                shapes = entry["shapes"]

                if self.label_filter is not None:
                    #filter annotation for labels
                    labels = [label["name"] for label in entry["labels"] if label["name"] in self.label_filter]
                    if len(labels) == 0:
                        #no labels match the filter
                        continue
                else:
                    labels = [label["name"] for label in entry["labels"]]
                labels = [self.replace_empty_with_no_object(label) for label in labels]

                new_label_ids = []
                for label in labels:
                    new_label_ids.append(label_name_to_id_mapping[label])

                if len(new_label_ids) == 0:
                    continue

                if shapes[0]["type"] == "point":
                    geometry = shapes[0]["geometry"]["points"][0]
                    radius = geometry["r"]
                    x, y = geometry["x"], geometry["y"]
                    new_shape = self._get_new_shape(x=x, y=y, radius=radius)
                elif shapes[0]["type"] == "polygon":
                    new_shape = self._get_new_polygon(
                        points=shapes[0]["geometry"]["points"]
                    )
                elif shapes[0]["type"] == "rect":
                    geometry = shapes[0]["geometry"]
                    x, y, w, h = geometry["x"], geometry["y"], geometry["width"], geometry["height"]
                    new_shape = self._get_new_rect(x=x, y=y, width=w, height=h)
                else:
                    raise ValueError(
                        f"Unsupported shape of type {shapes[0]['type']} found in "
                        f"annotation source data."
                    )
                sc_annotation = AnnotationRESTConverter.annotation_from_dict(
                    {
                        "shape": new_shape,
                        "labels": [
                            {"id": label_id, "probability": 1.0}
                            for label_id in new_label_ids
                        ]
                    }
                )
                annotation_list.append(sc_annotation)
        return annotation_list
