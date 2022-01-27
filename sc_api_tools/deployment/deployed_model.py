import json
import os
import shutil
import tempfile
import zipfile
from typing import Optional, Union, Dict, Tuple, Any

import attr

import numpy as np

from sc_api_tools.data_models import (
    OptimizedModel,
    TaskConfiguration
)
from sc_api_tools.data_models.enums import OpenvinoModelName
from sc_api_tools.http_session import SCSession
from sc_api_tools.rest_converters import ConfigurationRESTConverter, ModelRESTConverter


@attr.s(auto_attribs=True)
class DeployedModel(OptimizedModel):
    """
    This class represents an SC model that has been deployed for inference. It can be
    loaded onto a device to generate predictions
    """
    hyper_parameters: TaskConfiguration = attr.ib(kw_only=True, repr=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self._model_data_path: Optional[str] = None
        self._needs_tempdir_deletion: bool = False

    def get_data(self, source: Union[str, os.PathLike, SCSession]):
        """
        Loads the model weights from a data source. The `source` can be one of the
        following:

          1. The SC cluster (if an SCSession instance is passed). In this case the
             weights will be downloaded, and extracted to a temporary directory
          2. A zip file on local disk, in this case the weights will be extracted to a
             temporary directory
          3. A folder on local disk containing the .xml and .bin file for the model

        :param source: Data source to load the weights from
        """
        if isinstance(source, (os.PathLike, str)):
            if os.path.isfile(source) and os.path.splitext(source)[1] == '.zip':
                if self._model_data_path is None:
                    model_dir = tempfile.mkdtemp()
                    self._needs_tempdir_deletion = True
                else:
                    model_dir = self._model_data_path
                with zipfile.ZipFile(source, 'r') as zipped_source_model:
                    zipped_source_model.extractall(model_dir)
                self._model_data_path = os.path.join(model_dir, 'model')
                self.get_data(self._model_data_path)
            elif os.path.isdir(source):
                if 'model' in os.listdir(source):
                    source = os.path.join(source, 'model')
                source_contents = os.listdir(source)
                if 'model.bin' in source_contents and 'model.xml' in source_contents:
                    self._model_data_path = source
                else:
                    raise ValueError(
                        f"Unable to load model data from path '{source}'. Model "
                        f"file 'model.xml' and weights file 'model.bin' were not found "
                        f"at the path specified. "
                    )

        elif isinstance(source, SCSession):
            if self.base_url is None:
                raise ValueError(
                    f"Insufficient data to retrieve data for model {self}. Please set "
                    f"a base_url for the model first."
                )
            response = source.get_rest_response(
                url=self.base_url+'/export',
                method="GET",
                contenttype="zip"
            )
            filename = f"{self.name}_{self.optimization_type}_optimized.zip"
            model_dir = tempfile.mkdtemp()
            model_filepath = os.path.join(model_dir, filename)
            with open(model_filepath, 'wb') as f:
                f.write(response.content)
            self._model_data_path = model_dir
            self._needs_tempdir_deletion = True
            self.get_data(source=model_filepath)

    def __del__(self):
        """
        This method is called when the OptimizedModel object is deleted. It cleans up
        the temporary directory created to store the model data (if any)

        """
        if self._needs_tempdir_deletion:
            if os.path.exists(self._model_data_path):
                shutil.rmtree(os.path.dirname(self._model_data_path))

    def load_inference_model(
            self, model_name: OpenvinoModelName, device: str = 'CPU'
    ):
        """
        Loads the actual model weights to a specified device.

        :return: OpenVino inference engine model that can be used to make predictions
            on images
        """
        try:
            from openvino.model_zoo.model_api.models import Model as OMZModel
            from openvino.model_zoo.model_api.adapters import create_core, \
                OpenvinoAdapter
        except ImportError as error:
            raise ValueError(
                f"Unable to load inference model for {self}. Relevant OpenVINO "
                f"packages were not found. Please make sure that all packages from the "
                f"file `requirements-deployment.txt` have been installed. "
            ) from error

        model_adapter = OpenvinoAdapter(
            create_core(),
            model_path=os.path.join(self._model_data_path, 'model.xml'),
            weights_path=os.path.join(self._model_data_path, 'model.bin'),
            device=device,
            plugin_config=None,
            max_num_requests=1
        )
        model = OMZModel.create_model(
            name=str(model_name),
            model_adapter=model_adapter,
            configuration=None,
            preload=True
        )
        self._inference_model = model

    @classmethod
    def from_model_and_hypers(
            cls, model: OptimizedModel, hyper_parameters: TaskConfiguration
    ) -> 'DeployedModel':
        """
        Creates a DeployedModel instance out of an OptimizedModel and it's
        corresponding set of hyper parameters

        :param model: OptimizedModel to convert to a DeployedModel
        :param hyper_parameters: TaskConfiguration instance containing the hyper
            parameters for the model
        :return: DeployedModel instance
        """
        model_dict = model.to_dict()
        model_dict.update({'hyper_parameters': hyper_parameters})
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
    def from_folder(cls, path_to_folder: Union[str, os.PathLike]) -> 'DeployedModel':
        """
        Creates a DeployedModel instance from a folder containing the model data

        :param path_to_folder: Path to the folder that holds the model data
        :return: DeployedModel instance
        """
        config_filepath = os.path.join(path_to_folder, 'hyper_parameters.json')
        with open(config_filepath, 'r') as config_file:
            config_dict = json.load(config_file)
        hparams = ConfigurationRESTConverter.task_configuration_from_dict(config_dict)
        model_detail_path = os.path.join(path_to_folder, 'model.json')
        with open(model_detail_path, 'r') as model_detail_file:
            model_detail_dict = json.load(model_detail_file)
        model = ModelRESTConverter.optimized_model_from_dict(model_detail_dict)
        deployed_model = cls.from_model_and_hypers(
            model=model, hyper_parameters=hparams
        )
        deployed_model.get_data(source=path_to_folder)
        return deployed_model

    def save(self, path_to_folder: Union[str, os.PathLike]):
        """
        Saves the DeployedModel instance to the designated folder

        :param path_to_folder: Path to the folder to save the model to
        """
        if self._model_data_path is None:
            raise ValueError(
                "No model definition and model weights data was found for {self}, "
                "unable to save model."
            )
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)

        for model_file in os.listdir(self._model_data_path):
            shutil.copyfile(
                src=os.path.join(self._model_data_path, model_file),
                dst=os.path.join(path_to_folder, model_file)
            )

        config_dict = ConfigurationRESTConverter.configuration_to_minimal_dict(
            self.hyper_parameters
        )
        config_filepath = os.path.join(path_to_folder, 'hyper_parameters.json')
        with open(config_filepath, 'w') as config_file:
            json.dump(config_dict, config_file)

        model_detail_dict = self.to_dict()
        model_detail_dict.pop('hyper_parameters')
        model_detail_path = os.path.join(path_to_folder, 'model.json')
        with open(model_detail_path, 'w') as model_detail_file:
            json.dump(model_detail_dict, model_detail_file)
        return True

    def preprocess(
            self, image: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int, int]]]:
        """
        Preprocesses an image for inference

        :param image: Numpy array containing pixel data
        :return: Dictionary holding the preprocessing result, the original shape of
            the image and the shape after preprocessing
        """
        return self._inference_model.preprocess(image)

    def postprocess(
            self,
            inference_results: Dict[str, np.ndarray],
            metadata: Optional[Dict[str, Any]] = None
    ) -> Union[np.ndarray, 'openvino.model_zoo.model_api.models.utils.Detection']:
        """
        Postprocesses model outputs

        :param inference_results: Dictionary holding the results of inference
        :param metadata: Dictionary holding metadata
        :return: Postprocessed inference results. The exact format depends on the
            type of model that is loaded.
        """
        return self._inference_model.postprocess(inference_results, meta=metadata)

    def infer(
            self, preprocessed_image: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """

        :param preprocessed_image: Dictionary holding the preprocessing results for an
            image
        :return: Dictionary containing the model outputs
        """
        return self._inference_model.infer_sync(preprocessed_image)
