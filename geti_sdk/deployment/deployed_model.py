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

import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple, Union

import attr
import numpy as np
from openvino.model_api.adapters import OpenvinoAdapter, OVMSAdapter
from openvino.model_api.models import Model as model_api_Model
from openvino.runtime import Core

from geti_sdk.data_models import OptimizedModel, Project, TaskConfiguration
from geti_sdk.data_models.enums.domain import Domain
from geti_sdk.data_models.label import Label
from geti_sdk.data_models.label_group import LabelGroup, LabelGroupType
from geti_sdk.data_models.label_schema import LabelSchema
from geti_sdk.data_models.predictions import Prediction, ResultMedium
from geti_sdk.deployment.predictions_postprocessing.results_converter.results_to_prediction_converter import (
    ConverterFactory,
    InferenceResultsToPredictionConverter,
)
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_converters import ConfigurationRESTConverter, ModelRESTConverter

from .utils import (
    generate_ovms_model_address,
    generate_ovms_model_name,
    get_package_version_from_requirements,
    rgb_to_hex,
    target_device_is_ovms,
)

MODEL_DIR_NAME = "model"
PYTHON_DIR_NAME = "python"
WRAPPER_DIR_NAME = "model_wrappers"
REQUIREMENTS_FILE_NAME = "requirements.txt"

LABELS_CONFIG_KEY = "labels"
LABEL_TREE_KEY = "label_tree"
LABEL_GROUPS_KEY = "label_groups"
ALL_LABELS_KEY = "all_labels"

SALIENCY_KEY = "saliency_map"
ANOMALY_SALIENCY_KEY = "anomaly_map"
SEGMENTATION_SALIENCY_KEY = "soft_prediction"
FEATURE_VECTOR_KEY = "feature_vector"

OVMS_TIMEOUT = 10  # Max time to wait for OVMS models to become available


@attr.define
class DeployedModel(OptimizedModel):
    """
    Representation of an Intel® Geti™ model that has been deployed for inference. It
    can be loaded onto a device to generate predictions.
    """

    hyper_parameters: Optional[TaskConfiguration] = attr.field(
        kw_only=True, repr=False, default=None
    )

    def __attrs_post_init__(self):
        """
        Initialize private attributes
        """
        super().__attrs_post_init__()
        self._model_data_path: Optional[str] = None
        self._model_python_path: Optional[str] = None
        self._needs_tempdir_deletion: bool = False
        self._tempdir_path: Optional[str] = None
        self._has_custom_model_wrappers: bool = False
        self._label_schema: Optional[LabelSchema] = None

        # Attributes related to model explainability
        self._saliency_key: Optional[str] = None
        self._saliency_location: Optional[str] = None
        self._feature_vector_key: Optional[str] = None
        self._feature_vector_location: Optional[str] = None

        self._converter: Optional[Union[InferenceResultsToPredictionConverter]] = None

    @property
    def model_data_path(self) -> str:
        """
        Return the path to the raw model data

        :return: path to the directory containing the raw model data
        """
        if self._model_data_path is None:
            raise ValueError(
                "Model data path has not been set yet, location of binary model data "
                "is unknown."
            )
        return self._model_data_path

    def get_data(self, source: Union[str, os.PathLike, GetiSession]):
        """
        Load the model weights from a data source. The `source` can be one of the
        following:

          1. The Intel® Geti™ platform (if an GetiSession instance is passed). In this
            case the weights will be downloaded, and extracted to a temporary directory
          2. A zip file on local disk, in this case the weights will be extracted to a
             temporary directory
          3. A folder on local disk containing the .xml and .bin file for the model

        :param source: Data source to load the weights from
        """
        if isinstance(source, (os.PathLike, str)):
            if os.path.isfile(source) and os.path.splitext(source)[1] == ".zip":
                # Extract zipfile into temporary directory
                if self._model_data_path is None:
                    temp_dir = tempfile.mkdtemp()
                    self._needs_tempdir_deletion = True
                    self._tempdir_path = temp_dir
                else:
                    temp_dir = self._model_data_path

                with zipfile.ZipFile(source, "r") as zipped_source_model:
                    zipped_source_model.extractall(temp_dir)

                # _model_data_path contains the model structure and weights
                # _model_python_path contains the custom model wrappers
                self._model_data_path = os.path.join(temp_dir, MODEL_DIR_NAME)
                self._model_python_path = os.path.join(temp_dir, PYTHON_DIR_NAME)

                # Check if the model includes custom model wrappers
                if WRAPPER_DIR_NAME in os.listdir(self._model_python_path):
                    self._has_custom_model_wrappers = True
                self.get_data(temp_dir)

            elif os.path.isdir(source):
                source_contents = os.listdir(source)
                if MODEL_DIR_NAME in source_contents:
                    model_dir = os.path.join(source, MODEL_DIR_NAME)
                else:
                    model_dir = source
                model_dir_contents = os.listdir(model_dir)
                if (
                    "model.bin" in model_dir_contents
                    and "model.xml" in model_dir_contents
                ):
                    self._model_data_path = model_dir
                else:
                    raise ValueError(
                        f"Unable to load model data from path '{model_dir}'. Model "
                        f"file 'model.xml' and weights file 'model.bin' were not found "
                        f"at the path specified. "
                    )
                if PYTHON_DIR_NAME in source_contents:
                    model_python_path = os.path.join(source, PYTHON_DIR_NAME)
                else:
                    model_python_path = os.path.join(
                        os.path.dirname(source), PYTHON_DIR_NAME
                    )
                python_dir_contents = (
                    os.listdir(model_python_path)
                    if os.path.exists(model_python_path)
                    else []
                )
                if WRAPPER_DIR_NAME in python_dir_contents:
                    self._has_custom_model_wrappers = True

                self._model_python_path = os.path.join(source, PYTHON_DIR_NAME)

        elif isinstance(source, GetiSession):
            if self.base_url is None:
                raise ValueError(
                    f"Insufficient data to retrieve data for model {self}. Please set "
                    f"a base_url for the model first."
                )
            response = source.get_rest_response(
                url=self.base_url + "/export", method="GET", contenttype="zip"
            )
            filename = f"{self.name}_{self.optimization_type}_optimized.zip"
            model_dir = tempfile.mkdtemp()
            model_filepath = os.path.join(model_dir, filename)
            with open(model_filepath, "wb") as f:
                f.write(response.content)
            self._model_data_path = model_dir
            self._needs_tempdir_deletion = True
            self._tempdir_path = model_dir
            self.get_data(source=model_filepath)

    def __del__(self):
        """
        Clean up the temporary directory created to store the model data (if any). This
        method is called when the OptimizedModel object is deleted.
        """
        if self._needs_tempdir_deletion:
            if self._tempdir_path is not None and os.path.exists(self._tempdir_path):
                shutil.rmtree(self._tempdir_path)

    def load_inference_model(
        self,
        device: str = "CPU",
        configuration: Optional[Dict[str, Any]] = None,
        project: Optional[Project] = None,
    ) -> None:
        """
        Load the actual model weights to a specified device.

        :param device: Device (CPU or GPU) to load the model to. Defaults to 'CPU'
        :param configuration: Optional dictionary holding additional configuration
            parameters for the model
        :param project: Optional project to which the model belongs.
            This is only used when the model is run on OVMS, in that case the
            project is needed to identify the correct model
        :return: OpenVino inference engine model that can be used to make predictions
            on images
        """
        if not target_device_is_ovms(device=device):
            # Run the model locally
            model_adapter = OpenvinoAdapter(
                Core(),
                model=os.path.join(self._model_data_path, "model.xml"),
                weights_path=os.path.join(self._model_data_path, "model.bin"),
                device=device,
                plugin_config=None,
                max_num_requests=1,
            )
        else:
            # Connect to an OpenVINO model server instance
            model_name = generate_ovms_model_name(project=project, model=self)
            model_address = generate_ovms_model_address(
                ovms_address=device, model_name=model_name
            )

            ovms_connected = False
            ovms_error: Optional[BaseException] = None
            t_start = time.time()
            while not ovms_connected and time.time() - t_start < OVMS_TIMEOUT:
                # If OVMS has just started, model needs some time to initialize
                try:
                    model_adapter = OVMSAdapter(model_address)
                    ovms_connected = True
                except RuntimeError as error:
                    time.sleep(0.5)
                    ovms_error = error
            if not ovms_connected:
                if ovms_error is not None:
                    raise RuntimeError("Unable to connect to OVMS") from ovms_error
                else:
                    raise RuntimeError(
                        "Unknown error encountered while connecting to OVMS"
                    )

        # Load model configuration
        config_path = os.path.join(self._model_data_path, "config.json")
        if not os.path.isfile(config_path):
            raise ValueError(
                f"Missing configuration file `config.json` for deployed model `{self}`,"
                f" unable to load inference model."
            )
        with open(config_path, "r") as config_file:
            configuration_json = json.load(config_file)
        model_type = configuration_json.get("type_of_model")

        # Update model parameters
        parameters = configuration_json.get("model_parameters")
        if configuration is not None:
            configuration.update(parameters)
        else:
            configuration = parameters

        # Parse label schema
        label_dictionary = configuration_json.get("model_parameters").pop(
            LABELS_CONFIG_KEY, None
        )
        self._parse_label_schema_from_dict(label_dictionary)

        # Create model wrapper with the loaded configuration
        # First, add custom wrapper (if any) to path so that we can find it
        if self._has_custom_model_wrappers:
            wrapper_module_path = os.path.join(
                self._model_python_path, WRAPPER_DIR_NAME
            )
            module_name = WRAPPER_DIR_NAME + "." + model_type.lower().replace(" ", "-")
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, os.path.join(wrapper_module_path, "__init__.py")
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except ImportError as ex:
                raise ImportError(
                    f"Unable to load inference model for {self}. A custom model wrapper"
                    f"is required, but could not be found at path "
                    f"{wrapper_module_path}."
                ) from ex

        model = model_api_Model.create_model(
            model=model_adapter,
            model_type=model_type,
            preload=True,
            configuration=configuration,
        )
        self._inference_model = model

        # Load results to Prediction converter
        otx_version = get_package_version_from_requirements(
            requirements_path=os.path.join(
                self._model_python_path, REQUIREMENTS_FILE_NAME
            ),
            package_name="otx",
        )
        use_legacy_converter = not otx_version.startswith("1.5")
        self._converter = ConverterFactory.create_converter(
            self.label_schema, configuration, use_legacy_converter=use_leagacy_converter
        )

        # TODO: This is a workaround to fix the issue that causes the output blob name
        #  to be unset. Remove this once it has been fixed on OTX/ModelAPI side
        output_names = list(self._inference_model.outputs.keys())
        if hasattr(self._inference_model, "output_blob_name"):
            if not self._inference_model.output_blob_name:
                self._inference_model.output_blob_name = {
                    name: name for name in output_names
                }

    @classmethod
    def from_model_and_hypers(
        cls, model: OptimizedModel, hyper_parameters: Optional[TaskConfiguration] = None
    ) -> "DeployedModel":
        """
        Create a DeployedModel instance out of an OptimizedModel and it's
        corresponding set of hyper parameters.

        :param model: OptimizedModel to convert to a DeployedModel
        :param hyper_parameters: TaskConfiguration instance containing the hyper
            parameters for the model
        :return: DeployedModel instance
        """
        model_dict = model.to_dict()
        model_dict.update({"hyper_parameters": hyper_parameters})
        deployed_model = cls(**model_dict)
        try:
            model_group_id = model.model_group_id
            base_url = model.base_url
            deployed_model.model_group_id = model_group_id
            deployed_model.base_url = base_url
        except ValueError:
            pass
        return deployed_model

    @classmethod
    def from_folder(cls, path_to_folder: Union[str, os.PathLike]) -> "DeployedModel":
        """
        Create a DeployedModel instance from a folder containing the model data.

        :param path_to_folder: Path to the folder that holds the model data
        :return: DeployedModel instance
        """
        config_filepath = os.path.join(path_to_folder, "hyper_parameters.json")
        if os.path.isfile(config_filepath):
            with open(config_filepath, "r") as config_file:
                config_dict = json.load(config_file)
            hparams = ConfigurationRESTConverter.task_configuration_from_dict(
                config_dict
            )
        else:
            hparams = None
        model_detail_path = os.path.join(path_to_folder, "model.json")
        with open(model_detail_path, "r") as model_detail_file:
            model_detail_dict = json.load(model_detail_file)
        model = ModelRESTConverter.optimized_model_from_dict(model_detail_dict)
        deployed_model = cls.from_model_and_hypers(
            model=model, hyper_parameters=hparams
        )
        deployed_model.get_data(source=path_to_folder)
        return deployed_model

    def save(self, path_to_folder: Union[str, os.PathLike]) -> bool:
        """
        Save the DeployedModel instance to the designated folder.

        :param path_to_folder: Path to the folder to save the model to
        :return: True if the model was saved successfully, False otherwise
        """
        if self._model_data_path is None:
            raise ValueError(
                "No model definition and model weights data was found for {self}, "
                "unable to save model."
            )
        os.makedirs(path_to_folder, exist_ok=True, mode=0o770)

        new_model_data_path = os.path.join(path_to_folder, MODEL_DIR_NAME)
        new_model_python_path = os.path.join(path_to_folder, PYTHON_DIR_NAME)

        if (
            new_model_python_path == self._model_python_path
            or new_model_data_path == self._model_data_path
        ):
            logging.warning(
                f"Model '{self.name}' already exist in target path {path_to_folder}, "
                f"please save to a different location."
            )
            return False

        shutil.copytree(
            src=self._model_data_path,
            dst=new_model_data_path,
            dirs_exist_ok=True,
        )
        if self._model_python_path is not None:
            shutil.copytree(
                src=self._model_python_path,
                dst=new_model_python_path,
                dirs_exist_ok=True,
            )
            self._model_python_path = new_model_python_path

        self._model_data_path = new_model_data_path

        config_dict = ConfigurationRESTConverter.configuration_to_minimal_dict(
            self.hyper_parameters
        )
        config_filepath = os.path.join(path_to_folder, "hyper_parameters.json")
        with open(config_filepath, "w") as config_file:
            json.dump(config_dict, config_file, indent=4)

        model_detail_dict = self.to_dict()
        model_detail_dict.pop("hyper_parameters")
        model_detail_path = os.path.join(path_to_folder, "model.json")
        with open(model_detail_path, "w") as model_detail_file:
            json.dump(model_detail_dict, model_detail_file, indent=4)
        return True

    def _preprocess(
        self, image: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int, int]]]:
        """
        Preprocess an image for inference.

        :param image: Numpy array containing pixel data. The image is expected to have
            dimensions [height x width x channels], with the channels in RGB order
        :return: Dictionary holding the preprocessing result, the original shape of
            the image and the shape after preprocessing
        """
        return self._inference_model.preprocess(image)

    def _postprocess(
        self,
        inference_results: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[np.ndarray, List[Tuple[int, float]], Any]:
        """
        Postprocess the model outputs.

        :param inference_results: Dictionary holding the results of inference
        :param metadata: Dictionary holding metadata
        :return: Postprocessed inference results. The exact format depends on the
            type of model that is loaded:

            * For segmentation models, it will be a numpy array holding the output mask
            * For classification models, it will be a list of tuples holding the
                output class index and class probability
            * For detection models, it will be an instance of
                `openvino.model_zoo.model_api.models.utils.Detection`, holding the
                bounding box output
        """
        return self._inference_model.postprocess(inference_results, metadata)

    def _postprocess_explain_outputs(
        self,
        inference_results: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess the model outputs to obtain saliency maps, feature vectors and
        active scores.

        :param inference_results: Dictionary holding the results of inference
        :param metadata: Dictionary holding metadata
        :return: Tuple containing postprocessed outputs, formatted as follows:
            - Numpy array containing the saliency map
            - Numpy array containing the feature vector
        """
        if self._saliency_location is None and self._feature_vector_location is None:
            # When running postprocessing for the first time, we need to determine the
            # key and location of the saliency map and feature vector.
            if hasattr(self._inference_model, "postprocess_aux_outputs"):
                (
                    _,
                    saliency_map,
                    repr_vector,
                    _,
                ) = self._inference_model.postprocess_aux_outputs(
                    inference_results, metadata
                )
                self._saliency_location = "aux"
                self._feature_vector_location = "aux"
            else:
                # Check all possible saliency map keys in outputs and metadata
                if SALIENCY_KEY in inference_results.keys():
                    saliency_map = inference_results[SALIENCY_KEY]
                    self._saliency_location = "output"
                    self._saliency_key = SALIENCY_KEY
                elif SALIENCY_KEY in metadata.keys():
                    saliency_map = metadata[SALIENCY_KEY]
                    self._saliency_location = "meta"
                    self._saliency_key = SALIENCY_KEY
                elif ANOMALY_SALIENCY_KEY in metadata.keys():
                    saliency_map = metadata[ANOMALY_SALIENCY_KEY]
                    self._saliency_location = "meta"
                    self._saliency_key = ANOMALY_SALIENCY_KEY
                elif SEGMENTATION_SALIENCY_KEY in metadata.keys():
                    saliency_map = metadata[SEGMENTATION_SALIENCY_KEY]
                    self._saliency_location = "meta"
                    self._saliency_key = SEGMENTATION_SALIENCY_KEY
                else:
                    logging.warning("No saliency map found in model output")
                    saliency_map = None

                # Check all possible feature vector keys in outputs and metadata
                if FEATURE_VECTOR_KEY in inference_results.keys():
                    repr_vector = inference_results[FEATURE_VECTOR_KEY]
                    self._feature_vector_location = "output"
                    self._feature_vector_key = FEATURE_VECTOR_KEY
                elif FEATURE_VECTOR_KEY in metadata.keys():
                    repr_vector = metadata[FEATURE_VECTOR_KEY]
                    self._feature_vector_location = "meta"
                    self._feature_vector_key = FEATURE_VECTOR_KEY
                else:
                    logging.warning("No feature vector found in model output")
                    repr_vector = None
        else:
            # If location of feature vector and saliency map are already known, we can
            # use them directly
            if self._saliency_location == "aux":
                (
                    _,
                    saliency_map,
                    repr_vector,
                    _,
                ) = self._inference_model.postprocess_aux_outputs(
                    inference_results, metadata
                )
                return saliency_map, repr_vector
            elif self._saliency_location == "meta":
                saliency_map = metadata[self._saliency_key]
            elif self._saliency_location == "output":
                saliency_map = inference_results[self._saliency_key]
            else:
                logging.warning("No saliency map found in model output")
                saliency_map = None
            if self._feature_vector_location == "meta":
                repr_vector = metadata[self._feature_vector_key]
            elif self._feature_vector_location == "output":
                repr_vector = inference_results[self._feature_vector_key]

            else:
                logging.warning("No feature vector found in model output")
                repr_vector = None
        return saliency_map, repr_vector

    def infer(self, image: np.ndarray, explain: bool = False) -> Prediction:
        """
        Run inference on an already preprocessed image.

        :param preprocessed_image: Dictionary holding the preprocessing results for an
            image
        :return: Dictionary containing the model outputs
        """
        preprocessed_image, metadata = self._preprocess(image)
        # metadata is a dict with keys 'original_shape' and 'resized_shape'
        inference_results: Dict[str, np.ndarray] = self._inference_model.infer_sync(
            preprocessed_image
        )
        postprocessing_results = self._postprocess(inference_results, metadata=metadata)

        prediction = self._converter.convert_to_prediction(
            postprocessing_results, image_shape=metadata["original_shape"]
        )

        # Add optional explainability outputs
        if explain:
            saliency_map, repr_vector = self._postprocess_explain_outputs(
                inference_results=inference_results, metadata=metadata
            )
            prediction.feature_vector = repr_vector
            result_medium = ResultMedium(name="saliency map", type="saliency map")
            result_medium.data = saliency_map
            prediction.maps = [result_medium]

        return prediction

    @property
    def label_schema(self) -> LabelSchema:
        """
        Return the LabelSchema for the model.

        This requires the inference model to be loaded, getting this property while
        inference models are not loaded will raise a ValueError

        :return: LabelSchema containing the SDK label schema for the model
        """
        if self._label_schema is None:
            raise ValueError(
                "Inference model is not loaded, unable to retrieve label schema. "
                "Please load inference model first."
            )
        return self._label_schema

    def _parse_label_schema_from_dict(
        self, label_schema_dict: Optional[Dict[str, Union[dict, List[dict]]]] = None
    ) -> None:
        """
        Parse the dictionary contained in the model `config.json` file, and
        generate an  LabelSchema from it.

        :param label_schema_dict: Dictionary containing the label schema information
            to parse
        """
        label_groups_list = label_schema_dict[LABEL_GROUPS_KEY]
        labels_dict = label_schema_dict[ALL_LABELS_KEY]
        for key, value in labels_dict.items():
            color_tuple = tuple(
                int(value["color"][key]) for key in ["red", "green", "blue"]
            )
            label_entity = Label(
                name=value["name"],
                color=rgb_to_hex(color_tuple),
                group=None,
                is_empty=value.get("is_empty", False),
                hotkey=value["hotkey"],
                domain=Domain[value["domain"]],
                id=value["_id"],
                is_anomalous=value.get("is_anomalous", False),
            )
            labels_dict[key] = label_entity
        label_groups: List[LabelGroup] = []
        for group_dict in label_groups_list:
            labels: List[Label] = [
                labels_dict[label_id] for label_id in group_dict["label_ids"]
            ]
            label_groups.append(
                LabelGroup(
                    id=group_dict["_id"],
                    name=group_dict["name"],
                    group_type=LabelGroupType[group_dict["relation_type"]],
                    labels=labels,
                )
            )
        self._label_schema = LabelSchema(label_groups=label_groups)
