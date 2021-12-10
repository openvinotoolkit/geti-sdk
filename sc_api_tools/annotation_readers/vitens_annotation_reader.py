import glob
import json
import os
from typing import Tuple, List, Dict, Any

from .base_annotation_reader import AnnotationReader


class VitensAnnotationReader(AnnotationReader):
    """
    Class to read annotations for the Vitens dataset
    """
    def __init__(
            self,
            base_data_folder: str,
            annotation_format: str = ".json",
            task_type: str = "segmentation"
    ):
        super().__init__(
            base_data_folder=base_data_folder,
            annotation_format=annotation_format,
            task_type=task_type
        )

    def get_all_label_names(self):
        print(f"Reading annotation files in folder {self.base_folder}...")
        unique_label_names = []
        for annotation_file in os.listdir(self.base_folder):
            with open(os.path.join(self.base_folder, annotation_file), 'r') as f:
                data = json.load(f)
            for entry in data["data"]:
                labels = [label["name"] for label in entry["labels"]]
                for label in labels:
                    if label not in unique_label_names:
                        unique_label_names.append(label)
        return unique_label_names

    @staticmethod
    def _convert_filename(filename: str) -> Tuple[str, str]:
        name_separator = '_'
        return filename.split(name_separator)[0], name_separator

    def _get_new_shape(self, x: float, y: float, radius: float) -> dict:
        """
        Converts a NOUS point shape into SC shape format, for the appropriate task type.

        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :param radius: radius of the point
        :return: dictionary containing the new SC shape data
        """
        if self.task_type == "segmentation":
            new_shape = {
                "type": "ELLIPSE",
                "width": 2 * radius,
                "height": 2 * radius,
                "x": x - radius,
                "y": y - radius
            }
        elif self.task_type == "detection":
            new_shape = {
                "type": "RECTANGLE",
                "x": x,
                "y": y,
                "width": 2 * radius,
                "height": 2 * radius
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
        if self.task_type == "segmentation":
            shape = {"type": "POLYGON", "points": points}
        elif self.task_type == "detection":
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

    def get_data(self, filename: str, label_name_to_id_mapping: dict):
        annotation_filename, separator_token = self._convert_filename(filename)
        annotation_files = [
            self._convert_filename(filename)[0] for filename
            in os.listdir(self.base_folder)
        ]
        filepath = glob.glob(
            os.path.join(self.base_folder, f"{annotation_filename}{separator_token}*")
        )
        annotation_list = []
        if annotation_filename not in annotation_files or len(filepath) == 0:
            print(f"No annotation file found for image {filename}.")
        else:
            if len(filepath) != 1:
                print(
                    f"Multiple matching annotation files found for image with "
                    f"name {annotation_filename}. Skipping this image..."
                )
                return []
            else:
                filepath = filepath[0]
            with open(filepath, 'r') as f:
                data = json.load(f)
            for entry in data["data"]:
                shapes = entry["shapes"]
                labels = [label["name"] for label in entry["labels"]]
                new_label_ids = [label_name_to_id_mapping[label] for label in labels]

                if shapes[0]["type"] == "point":
                    geometry = shapes[0]["geometry"]["points"][0]
                    radius = geometry["r"]
                    x, y = geometry["x"], geometry["y"]
                    new_shape = self._get_new_shape(x=x, y=y, radius=radius)
                elif shapes[0]["type"] == "polygon":
                    new_shape = self._get_new_polygon(
                        points=shapes[0]["geometry"]["points"]
                    )
                else:
                    raise ValueError(
                        f"Unsupported shape of type {shapes[0]['type']} found in "
                        f"annotation source data."
                    )
                annotation_list.append(
                    {
                        "shape": new_shape,
                        "labels": [{"id": label_id} for label_id in new_label_ids]
                    }
                )
        return annotation_list