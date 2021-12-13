import copy
import json
import os
import time
import warnings
from typing import Dict, Sequence, Generic, TypeVar, Any, List, Optional

from sc_api_tools.annotation_readers.base_annotation_reader import AnnotationReader
from sc_api_tools.data_models import Project
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
            project: Project,
            image_to_id_mapping: Dict[str, str],
            annotation_reader: Optional[AnnotationReaderType] = None
    ):
        self.session = session
        project_id = project.id
        dataset_id = project.datasets[0].id
        self.base_url = f"workspaces/{workspace_id}/projects/{project_id}/datasets/" \
                        f"{dataset_id}/media"
        self.image_id_to_name_mapping = {
            id_: name for name, id_ in image_to_id_mapping.items()
        }
        self.annotation_reader = annotation_reader
        self._project = project
        if annotation_reader is None:
            warnings.warn(
                "You did not specify an annotation reader for the annotation manager, "
                "this means it can only be used for annotation downloading, but not "
                "for uploading."
            )
            label_mapping = None
        else:
            label_mapping = self._get_label_mapping(project)
        self._label_mapping = label_mapping
        self._original_label_mapping = copy.deepcopy(self._label_mapping)

    def _get_label_mapping(self, project: Project) -> Dict[str, str]:
        """
        Get the mapping of the label names to the label ids for the project

        :param project:
        :return: Dictionary containing the label names as keys and the label ids as
            values
        """
        source_label_names = self.annotation_reader.get_all_label_names()
        project_label_mapping = project.pipeline.label_id_to_name_mapping
        project_label_name_to_id_mapping = {
            name: id_ for (id_, name) in project_label_mapping.items()
        }
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
        if self.annotation_reader is not None:
            if self._label_mapping is not None:
                return self._label_mapping
            else:
                self._label_mapping = self._get_label_mapping(self._project)
                self._original_label_mapping = copy.deepcopy(self._label_mapping)
        else:
            raise ValueError(
                "Unable to get label mapping for this annotation manager, no "
                "annotation reader has been defined."
            )

    def upload_annotation_for_image(self, image_id: str):
        """
        Uploads a new annotation for the image with id `image_id` to the cluster. This
        will overwrite any current annotations for the image.

        :param image_id:
        :return:
        """
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
        image_name = self.image_id_to_name_mapping[image_id]
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

    def download_annotations_for_images(
            self, image_ids: List[str], path_to_folder: str
    ) -> None:
        """
        Donwnloads annotations from the server to a target folder on disk

        :param image_ids: List of ids of images for which to download the annotations
        :param path_to_folder: Folder to save the annotations to
        """
        path_to_annotations_folder = os.path.join(path_to_folder, "annotations")
        if not os.path.exists(path_to_annotations_folder):
            os.makedirs(path_to_annotations_folder)
        print(
            f"Starting annotation download... saving annotations for "
            f"{len(image_ids)} to folder {path_to_annotations_folder}"
        )
        t_start = time.time()
        download_count = 0
        skip_count = 0
        for image_id in image_ids:
            image_name = self.image_id_to_name_mapping[image_id]
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

            annotation_path = os.path.join(
                path_to_annotations_folder, image_name + '.json'
            )
            with open(annotation_path, 'w') as f:
                json.dump(export_data, f)
            download_count += 1
        t_elapsed = time.time() - t_start
        if download_count > 0:
            msg = f"Downloaded {download_count} annotations to folder " \
                  f"{path_to_folder} in {t_elapsed:.1f} seconds."
        else:
            msg = f"No annotations were downloaded."
        if skip_count > 0:
            msg = msg + f" Was unable to retrieve annotations for {skip_count} " \
                        f"images, these images were skipped."
        print(msg)

    def download_all_annotations(self, path_to_folder: str) -> None:
        """
        Donwnloads all annotations for the project to a target folder on disk

        :param path_to_folder: Folder to save the annotations to
        """
        image_ids = list(self.image_id_to_name_mapping.keys())
        self.download_annotations_for_images(
            image_ids=image_ids, path_to_folder=path_to_folder
        )
