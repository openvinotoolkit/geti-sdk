import copy
import json
import os
import time
from typing import Dict, Sequence, Generic, TypeVar, List, Any

from sc_api_tools.annotation_readers.base_annotation_reader import AnnotationReader
from sc_api_tools.http_session import SCSession

AnnotationReaderType = TypeVar("AnnotationReaderType", bound=AnnotationReader)


class AnnotationManager(Generic[AnnotationReaderType]):
    """
    Class to upload annotations to an existing project
    """

    def __init__(
            self,
            session: SCSession,
            workspace_id: str,
            project: dict,
            image_to_id_mapping: Dict[str, str],
            annotation_reader: AnnotationReaderType
    ):
        self.session = session
        project_id = project["id"]
        dataset_id = project["datasets"][0]["id"]
        self.base_url = f"workspaces/{workspace_id}/projects/{project_id}/datasets/" \
                        f"{dataset_id}/media"
        self.id_to_image_name_mapping = {
            id_: name for name, id_ in image_to_id_mapping.items()
        }
        self.annotation_reader = annotation_reader

        self._label_mapping = self._get_label_mapping(project)
        self._original_label_mapping = copy.deepcopy(self.label_mapping)

    def _get_label_mapping(self, project: dict):
        source_label_names = self.annotation_reader.get_all_label_names()
        trainable_tasks = [
            task for task in project["pipeline"]["tasks"]
            if task["task_type"] not in ["dataset", "crop"]
        ]
        task_label_info = [task["labels"] for task in trainable_tasks]
        project_label_name_to_id_mapping: Dict[str, str] = {}
        for label_list in task_label_info:
            project_label_name_to_id_mapping.update(
                {label["name"]: label["id"] for label in label_list}
            )
        for source_label_name in source_label_names:
            if source_label_name not in project_label_name_to_id_mapping:
                raise ValueError(
                    f"Found label {source_label_name} in source labels, but this "
                    f"label is not in the project labels."
                )
        return project_label_name_to_id_mapping

    @property
    def label_mapping(self) -> Dict[str, str]:
        """
        Returns dictionary with label names as keys and label ids as values

        :return:
        """
        return self._label_mapping

    def upload_annotation_for_image(self, image_id: str):
        annotation_data = self._read_and_convert_annotation_for_image_from_source(
            image_id=image_id
        )
        if annotation_data["annotations"]:
            response = self.session.get_rest_response(
                url=f"{self.base_url}/images/{image_id}/annotations",
                method="POST",
                data=annotation_data
            )
        else:
            response = {}
        return response

    def get_latest_annotation_for_image(self, image_id: str) -> Dict[str, Any]:
        """
        Retrieve the latest annotation for an image with id `image_id` from the cluster

        :param image_id: ID of the image to retrieve the annotation for
        :return:
        """
        return self.session.get_rest_response(
            url=f"{self.base_url}/images/{image_id}/annotations/latest",
            method="GET"
        )

    def append_annotation_for_image(self, image_id: str):
        new_annotation_data = self._read_and_convert_annotation_for_image_from_source(
            image_id=image_id
        )
        try:
            existing_annotation = self.get_latest_annotation_for_image(image_id)
        except ValueError:
            print(f"No existing annotation found for image with id {image_id}")
            existing_annotation = {
                "annotations": [],
                "media_identifier": {"type": "image", "image_id": image_id}
            }
        annotation_data = existing_annotation["annotations"]
        for annotation in annotation_data:
            keys = copy.deepcopy(list(annotation.keys()))
            for key in keys:
                if key not in ["labels", "shape"]:
                    annotation.pop(key, None)
        annotation_data.extend(new_annotation_data["annotations"])
        request_data = {
            "media_identifier": existing_annotation["media_identifier"],
            "annotations": annotation_data
        }
        response = self.session.get_rest_response(
            url=f"{self.base_url}/images/{image_id}/annotations",
            method="POST",
            data=request_data
        )
        return response

    def _read_and_convert_annotation_for_image_from_source(self, image_id: str):
        image_name = self.id_to_image_name_mapping[image_id]
        annotation_list = self.annotation_reader.get_data(
            filename=image_name, label_name_to_id_mapping=self.label_mapping
        )
        return {
            "media_identifier": {
                "type": "image",
                "image_id": image_id
            },
            "annotations": annotation_list,
        }

    def upload_annotations_for_images(
            self, image_id_list: Sequence[str], append_annotations: bool = False
    ):
        """
        Uploads annotations for a list of images. If append_annotations is set to True,
        annotations will be appended to the existing annotations for the image in the
        project. If set to False, existing annotations will be overwritten.

        :param image_id_list:
        :param append_annotations:
        :return:
        """
        print("Starting annotation upload...")
        upload_count = 0
        for image_id in image_id_list:
            if not append_annotations:
                response = self.upload_annotation_for_image(image_id=image_id)
            else:
                response = self.append_annotation_for_image(image_id=image_id)
            if response:
                upload_count += 1
        if upload_count > 0:
            print(f"Upload complete. Uploaded {upload_count} new annotations")
        else:
            print(
                "No new annotations were found.")

    def download_and_save_annotations_for_images(
            self, image_id_list: Sequence[str], path_to_target_folder: str
    ) -> None:
        """
        Donwnloads annotations from the server to a target folder on disk

        :param image_id_list:
        :param path_to_target_folder:
        """
        print(
            f"Starting annotation download... saving to folder {path_to_target_folder}"
        )
        if not os.path.exists(path_to_target_folder):
            os.makedirs(path_to_target_folder)
        t_start = time.time()
        download_count = 0
        skip_count = 0
        for image_id in image_id_list:
            image_name = self.id_to_image_name_mapping[image_id]
            try:
                annotation_data = self.get_latest_annotation_for_image(image_id)
            except ValueError:
                print(
                    f"Unable to retrieve latest annotation for image {image_name}. "
                    f"Skipping this image"
                )
                skip_count += 1
                continue

            kind = annotation_data.pop("kind", None)
            if kind != "annotation":
                print(
                    f"Received invalid annotation of kind {kind} for image{image_name}"
                )
                skip_count += 1
                continue
            annotation_data.pop("modified", None)
            annotation_data.pop("media_identifier", None)
            annotation_data.pop("id", None)
            export_data = copy.deepcopy(annotation_data)
            for annotation in export_data["annotations"]:
                annotation.pop("id", None)
                annotation.pop("modified", None)
                for label in annotation["labels"]:
                    label.pop("id")

            image_path = os.path.join(path_to_target_folder, image_name+'.json')
            with open(image_path, 'w') as f:
                json.dump(export_data, f)
            download_count += 1
        t_elapsed = time.time() - t_start
        if download_count > 0:
            msg = f"Downloaded {download_count} annotations to foler " \
                  f"{path_to_target_folder} in {t_elapsed:.1f} seconds."
        else:
            msg = f"No annotations were donwloaded."
        if skip_count > 0:
            msg = msg + f" Was unable to retrieve annotations for {skip_count} " \
                        f"images, these images were skipped."
        print(msg)
