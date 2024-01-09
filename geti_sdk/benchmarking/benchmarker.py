# Copyright (C) 2023 Intel Corporation
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
import csv
import itertools
import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geti_sdk import Geti
from geti_sdk.data_models import (
    Image,
    Model,
    OptimizedModel,
    Performance,
    Project,
    Video,
)
from geti_sdk.deployment import Deployment
from geti_sdk.rest_clients import ImageClient, ModelClient, TrainingClient, VideoClient

from .utils import get_system_info, load_benchmark_media, suppress_log_output


class Benchmarker:
    """
    Initialize and manage benchmarking experiments to measure model throughput on
    different hardware.
    """

    def __init__(
        self,
        geti: Geti,
        project: Union[str, Project],
        precision_levels: Optional[Sequence[str]] = None,
        models: Optional[Sequence[Model]] = None,
        algorithms: Optional[Sequence[str]] = None,
        benchmark_images: Optional[
            Sequence[Union[Image, np.ndarray, os.PathLike]]
        ] = None,
        benchmark_video: Optional[Union[Video, os.PathLike]] = None,
    ):
        """
        Manage benchmarking experiments to measure inference model throughput on
        different hardware. It allows for quick and easy comparison of inference
        framerates for different model architectures and precision levels for the
        specified project.

        The Benchmarker will fetch models from Intel Geti and measure the
        throughput for local inference for these models. It requires an existing Geti
        Project to work. The project does not need to be trained yet, but must have
        sufficient annotations to be able to start training.

        The benchmark can be run on any target device that is supported by OpenVINO.
        Specifying multiple algorithms and precision levels to include in the
        benchmark allows for easy comparison of the performance for these models.

        NOTE: For task chain projects, passing `models` or `algorithms` directly in the
        constructor is not supported. The Benchmarker provides dedicated
        methods to specify which models or algorithms to use in this case, which can
        be called after initialization.

        :param geti: Geti instance on which the project to use for benchmarking lives
        :param project: Project or project name to use for the benchmarking. The
            project must exist on the specified Geti instance
        :param precision_levels: List of model precision levels to run the
            benchmarking for. Throughput will be measured for each precision level
            in this list. Valid options are ["FP32", "FP16", "INT8"]. Defaults to:
            ["FP32", "FP16"]
        :param models: Optional list of base models to use in the benchmarking.
            Note that this should not be optimized models: The Benchmarker will fetch
            optimized models in the specified `precision_levels` for the base
            models specified here
        :param algorithms: List of algorithm names for which the benchmarking should
            be performed. If no models are trained for all algorithms in the list,
            the Benchmarker will trigger training for those algorithms and wait for
            the training jobs to complete. If neither `models` or `algorithms` are
            specified, the Benchmarker uses the current active model(s) in the project.
        :param benchmark_images: Optional list of images to run the benchmarks on. Can
            be passed as Geti Images, numpy images or filepaths to image files on disk.
            NOTE: If no images are specified, the benchmark will use a random image
            or video frame from the project
        :param benchmark_video: Optional video to use for the benchmarking. Can be
            passed either as a Geti Video object, or as a filepath to a video file on
            disk.
            NOTE: Either `benchmark_video` or `benchmark_images` can be specified, but
            not both. If neither images nor video is passed, the Benchmarker will
            pick a random image or video frame from the project to run the benchmark
            on.
        """
        self.geti = geti
        if isinstance(project, str):
            project_name = project
        else:
            project_name = project.name
        self.project = geti.get_project(project_name)
        logging.info(
            f"Setting up Benchmarker for Intel® Geti™ project `{self.project.name}`."
        )
        self._is_single_task = len(self.project.get_trainable_tasks()) == 1
        if precision_levels is None:
            precision_levels = ["FP32", "FP16"]
        self.precision_levels = precision_levels
        self.model_client = ModelClient(
            session=geti.session, workspace_id=geti.workspace_id, project=self.project
        )
        self.training_client = TrainingClient(
            session=geti.session, workspace_id=geti.workspace_id, project=self.project
        )

        self._models: Optional[List[Model]] = None
        self._task_chain_models: Optional[List[List[Model]]] = None
        self._algorithms: Optional[List[str]] = None
        self._task_chain_algorithms: Optional[List[List[str]]] = None
        self._optimized_models: Optional[List[OptimizedModel]] = None
        self._task_chain_optimized_models: Optional[List[List[OptimizedModel]]] = None
        self._deployment_folders: List[os.PathLike] = []

        if not self._is_single_task and (models is not None or algorithms is not None):
            raise ValueError(
                "You have specified a task-chain project to be used for benchmarking. "
                "This does not allow setting `models` or `algorithms` via the "
                "Benchmark constructor. Please specify which models or algorithms "
                "should be used in the benchmark using the methods "
                "`Benchmarker.set_task_chain_models()` or "
                "`Benchmarker.set_task_chain_algorithms()`. These methods "
                "can be used after initialization of the Benchmarker. "
                "Please refer to the method documentation for further guidance."
            )

        if models is None and algorithms is None:
            if self._is_single_task:
                logging.info(
                    "No `models` or `algorithms` were specified, using current active "
                    "models in the project."
                )
            else:
                logging.info(
                    "Models or algorithms to benchmark for a task chain project can "
                    "be specified by calling the `Benchmarker.set_task_chain_models()`"
                    " or `Benchmarker.set_task_chain_algorithms()` methods. If these "
                    "methods are not called, the benchmark will be set up using the "
                    "current active models."
                )
            models = self.model_client.get_all_active_models()
            if len(models) == 0:
                raise ValueError(
                    "No trained models were found in the project, please either "
                    "train a model first or specify an algorithm to train."
                )
            algorithms: List[str] = []
            for model in models:
                task = self.model_client.get_task_for_model(model=model)
                if self._is_single_task:
                    logging.info(
                        f"Found active model `{model.name}` for task `{task.title}`"
                    )
                algorithms.append(model.architecture)
            if self._is_single_task:
                self._models = models
                self._algorithms = algorithms
            else:
                self._task_chain_models = [[model] for model in models]
                self._task_chain_algorithms = [[algo] for algo in algorithms]
            self._are_models_specified = True

        elif models is not None and algorithms is not None:
            raise ValueError(
                "Either `models` or `algorithms` could be specified, but not both."
            )
        elif models is not None and algorithms is None:
            self._models = models
            self._are_models_specified = True
            self._algorithms = [model.architecture for model in models]
        elif models is None and algorithms is not None:
            self._models = None
            self._are_models_specified = False
            self._algorithms = algorithms

        self.images = None
        self.video = None
        if benchmark_video is not None and benchmark_images is not None:
            raise ValueError(
                "Please specify either `benchmark_video` or `benchmark_images`, but "
                "not both."
            )
        elif benchmark_images is not None:
            self.images = benchmark_images
        elif benchmark_video is not None:
            self.video = benchmark_video
        else:
            image_client = ImageClient(
                session=geti.session,
                workspace_id=geti.workspace_id,
                project=self.project,
            )
            images = image_client.get_all_images()
            if len(images) > 0:
                self.images = [images[0]]
            else:
                video_client = VideoClient(
                    session=geti.session,
                    workspace_id=geti.workspace_id,
                    project=self.project,
                )
                videos = video_client.get_all_videos()
                if len(videos) > 0:
                    self.video = videos[0]
                else:
                    raise ValueError(
                        "No benchmark images or video was specified, and no media "
                        "could be found in the project. Unable to proceed, please "
                        "specify either images or a video to run the benchmark on."
                    )

    def set_task_chain_models(
        self, models_task_1: Sequence[Model], models_task_2: Sequence[Model]
    ):
        """
        Set the models to be used in the benchmark for a task-chain project. The
        benchmarking will run for all possible combinations of models for task 1
        and task 2.

        :param models_task_1: Models to use for task #1
        :param models_task_2: Models to use for task #2
        """
        if self._is_single_task:
            logging.warning(
                "Method `set_task_chain_models` was called for a non-task-chain "
                "project. This has no effect."
            )
            return
        self._task_chain_models = [
            list(pair) for pair in itertools.product(models_task_1, models_task_2)
        ]
        self._are_models_specified = True
        self._task_chain_algorithms = [
            [model.architecture for model in pair] for pair in self._task_chain_models
        ]
        logging.info(
            f"Task chain models set. Found a total of "
            f"{len(self._task_chain_models)} possible combinations to benchmark"
        )

    def set_task_chain_algorithms(
        self, algorithms_task_1: Sequence[str], algorithms_task_2: Sequence[str]
    ):
        """
        Set the algorithms to be used in the benchmark for a task-chain project. The
        benchmarking will run for all possible combinations of algorithms for task 1
        and task 2.

        Note that upon benchmark initialization, the Benchmarker will check if
        trained models are available for all algorithms specified

        :param algorithms_task_1: Algorithms to use for task #1
        :param algorithms_task_2: Algorithms to use for task #2
        """
        if self._is_single_task:
            logging.warning(
                "Method `set_task_chain_algorithms` was called for a non-task-chain "
                "project. This has no effect."
            )
            return
        self._task_chain_algorithms = [
            list(pair)
            for pair in itertools.product(algorithms_task_1, algorithms_task_2)
        ]
        self._are_models_specified = False
        self._task_chain_models = None
        logging.info(
            f"Task chain algorithms set. Found a total of "
            f"{len(self._task_chain_algorithms)} possible algorithm combinations "
            f"to benchmark"
        )

    @property
    def models(self) -> Union[List[Model], List[List[Model]]]:
        """
        Return the models to be used in the benchmark.
        """
        if self._are_models_specified:
            if self._is_single_task:
                return self._models
            return self._task_chain_models
        raise ValueError(
            "Unable to access models, no benchmark models have been specified yet."
        )

    @property
    def algorithms(self) -> Union[List[str], List[List[str]]]:
        """
        Return the algorithm names to be used in the benchmark
        """
        if self._is_single_task:
            return self._algorithms
        return self._task_chain_algorithms

    @property
    def optimized_models(
        self,
    ) -> Union[List[OptimizedModel], List[List[OptimizedModel]]]:
        """
        Return the optimized models to be used in deployments for the benchmark
        """
        if self._is_single_task and self._optimized_models is not None:
            return self._optimized_models
        elif not self._is_single_task and self._task_chain_optimized_models is not None:
            return self._task_chain_optimized_models
        raise ValueError(
            "Optimized models have not been assigned yet. Please initialize the "
            "benchmarker first using the `initialize_benchmark` method."
        )

    def _train_model_for_algorithm(self, task_index: int, algorithm_name: str) -> Model:
        """
        Train a model for the specified task and algorithm

        :param task_index: Index of the task for which to train the model
        :param algorithm_name: Name of the algorithm to use
        :return: Model object, representing the trained model
        """
        algos = self.training_client.get_algorithms_for_task(task_index)
        algo = algos.get_by_name(algorithm_name)
        job = self.training_client.train_task(task_index, algorithm=algo)
        job = self.training_client.monitor_job(job=job)
        return self.model_client.get_model_for_job(job)

    def _optimize_model_for_algorithm(
        self, model: Model, precision: str = "INT8"
    ) -> OptimizedModel:
        """
        Optimize a `model` with the given `precision`.

        :param model: Base model to optimize
        :param precision: Precision to use for the optimization
        :return: OptimizedModel object, representing the optimized model
        """
        optimization_type_map = {"INT8": "pot"}
        job = self.model_client.optimize_model(
            model=model, optimization_type=optimization_type_map[precision]
        )
        self.model_client.monitor_job(job)
        updated_model = self.model_client.update_model_detail(model)
        return updated_model.get_optimized_model(
            precision=precision, optimization_type="openvino"
        )

    def prepare_benchmark(self, working_directory: os.PathLike = "."):
        """
        Prepare the benchmarking experiments. This involves:

            1. Ensuring that all required models are available, i.e. that all
                specified algorithms have a trained model in the Geti project. If
                not, training jobs will be started and awaited.
            2. Ensuring that for each model, optimized models with the required
                quantization level are available. If not, optimization jobs will
                be started and awaited.
            3. Creating and downloading deployments for all models to benchmark.

        :param working_directory: Output directory to which the deployments for the
            benchmark will be saved.
        """
        logging.info("Preparing benchmark experiments.")
        # First, check if model training is required. This happens when
        # algorithms are specified, but not all algorithms may have a model
        # trained already.
        if not self._are_models_specified:
            if self._is_single_task:
                logging.info(
                    f"Checking model availability for {len(self.algorithms)} "
                    f"different algorithms: {self.algorithms}."
                )
                models: List[Model] = []
                for algorithm_name in self.algorithms:
                    model = self.model_client.get_latest_model_by_algo_name(
                        algorithm_name
                    )
                    if model is None:
                        logging.info(
                            f"No model found in project for algorithm "
                            f"{algorithm_name}, requesting model training"
                        )
                        model = self._train_model_for_algorithm(
                            task_index=0, algorithm_name=algorithm_name
                        )
                    models.append(model)
                self._models = models
            else:
                logging.info(
                    f"Checking model availability for {len(self.algorithms)} "
                    f"different pairs of algorithms."
                )
                models: List[List[Model]] = []
                for algo_pair in self.algorithms:
                    model_pair: List[Model] = []
                    for task_index, algorithm_name in enumerate(algo_pair):
                        model = self.model_client.get_latest_model_by_algo_name(
                            algorithm_name
                        )
                        if model is None:
                            model = self._train_model_for_algorithm(
                                task_index=task_index, algorithm_name=algorithm_name
                            )
                        model_pair.append(model)
                    models.append(model_pair)
                self._task_chain_models = models
            self._are_models_specified = True
        logging.info("All required base models are available")
        # Then, check if model optimization is required. Create a list of
        # optimized models for which deployments should be created.
        if self._is_single_task:
            logging.info(
                f"Checking optimized model availability for {len(self.models)} "
                f"different models and quantization levels: {self.precision_levels}"
            )
            optimized_models: List[OptimizedModel] = []
            for model in self.models:
                for precision in self.precision_levels:
                    opt_model = model.get_optimized_model(
                        precision=precision, optimization_type="openvino"
                    )
                    if opt_model is None:
                        logging.info(
                            f"No optimized model with quantization level {precision} "
                            f"found for model {model.name}, requesting model "
                            f"optimization."
                        )
                        opt_model = self._optimize_model_for_algorithm(
                            model=model, precision=precision
                        )
                    optimized_models.append(opt_model)
            self._optimized_models = optimized_models
        else:
            logging.info(
                f"Checking optimized model availability for {len(self.models)} "
                f"different model pairs and quantization levels: "
                f"{self.precision_levels}"
            )
            optimized_models: List[List[OptimizedModel]] = []
            for model_pair in self.models:
                for precision in self.precision_levels:
                    opt_model_pair: List[OptimizedModel] = []
                    for model in model_pair:
                        # Update model info, to take into account any
                        # optimizations made during loop execution
                        model = self.model_client.update_model_detail(model=model)
                        opt_model = model.get_optimized_model(
                            precision=precision, optimization_type="openvino"
                        )
                        if opt_model is None:
                            logging.info(
                                f"No optimized model with quantization level {precision} "
                                f"found for model {model.name}, requesting model "
                                f"optimization."
                            )
                            opt_model = self._optimize_model_for_algorithm(
                                model=model, precision=precision
                            )
                        opt_model_pair.append(opt_model)
                    optimized_models.append(opt_model_pair)
            self._task_chain_optimized_models = optimized_models
        logging.info("All required optimized models are available")

        # Next up, create deployments for all optimized models
        os.makedirs(working_directory, exist_ok=True)
        logging.info(
            f"Creating {len(self.optimized_models)} deployments to benchmark. Saving "
            f"deployment data to folder: `{working_directory}`"
        )
        with logging_redirect_tqdm(tqdm_class=tqdm):
            for index, opt_models in tqdm(
                enumerate(self.optimized_models),
                total=len(self.optimized_models),
                desc="Creating deployments",
            ):
                if isinstance(opt_models, OptimizedModel):
                    opt_models = [opt_models]
                output_folder = os.path.join(working_directory, f"deployment_{index}")
                with suppress_log_output():
                    self.geti.deploy_project(
                        project_name=self.project.name,
                        output_folder=output_folder,
                        models=opt_models,
                    )
                    self._deployment_folders.append(output_folder)
        logging.info("Deployments created. Benchmark initialization complete.")

    def initialize_from_folder(self, target_folder: os.PathLike = "."):
        """
        Initialize the Benchmarker from a folder containing deployments. This method
        checks that any directory inside the `target_folder` contains a valid
        deployment for the project assigned to this Benchmarker.

        :param target_folder: Directory containing the model deployments that should
            be used in the Benchmarking.
        """
        logging.info(f"Initializing Benchmarker from folder `{target_folder}`")
        if len(self._deployment_folders) != 0:
            raise ValueError(
                "The Benchmarker appears to be already initialized. Unable to "
                "continue."
            )
        for object in os.listdir(target_folder):
            object_path = os.path.join(target_folder, object)
            if os.path.isfile(object_path):
                continue
            try:
                deployment = Deployment.from_folder(path_to_folder=object_path)
            except ValueError:
                logging.info(
                    f"Directory {object_path} does not contain a valid Deployment, "
                    f"skipping..."
                )
                continue
            if deployment.project.name != self.project.name:
                raise ValueError(
                    f"Deployment at path `{object_path}` was created for project "
                    f"{deployment.project.name}, yet the Benchmarker was set up for "
                    f"project {self.project.name}. Unable to initialize. Please use "
                    f"only deployments created for project `{self.project.name}`"
                )
            self._deployment_folders.append(object_path)

        if len(self._deployment_folders) != 0:
            logging.info(
                f"Benchmarker was initialized from folder. {len(self._deployment_folders)} "
                f"valid deployment folders were found."
            )

    def run_throughput_benchmark(
        self,
        working_directory: os.PathLike = ".",
        results_filename: str = "results",
        target_device: str = "CPU",
        frames: int = 200,
        repeats: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Run the benchmark experiment.

        :param working_directory: Directory in which the deployments that should be
            benchmarked are stored. All output will be saved to this directory.
        :param results_filename: Name of the file to which the results will be saved.
            File extension should not be included, the results will always be saved as
            a `.csv` file. Defaults to `results.csv`. The results file will be created
            within the `working_directory`
        :param target_device: Device to run the inference models on, for example "CPU"
            or "GPU". Defaults to "CPU".
        :param frames: Number of frames/images to infer in order to calculate
            fps
        :param repeats: Number of times to repeat the benchmark runs. FPS will be
            averaged over the runs.
        """
        if len(self._deployment_folders) == 0:
            raise ValueError(
                "Benchmarker does not contain any deployments to benchmark yet! Please "
                "prepare the deployments first using either the "
                "`Benchmarker.prepare_benchmark()` or "
                "`Benchmarker.initialize_from_folder()` methods."
            )
        logging.info("Starting throughput benchmark experiments.")

        logging.info(
            f"The Benchmarker will run for {len(self._deployment_folders)} deployments"
        )

        logging.info("Loading benchmark media")
        benchmark_frames = load_benchmark_media(
            session=self.geti.session,
            images=self.images,
            video=self.video,
            frames=frames,
        )

        results_file = os.path.join(working_directory, f"{results_filename}.csv")
        logging.info(f"Writing results to `{results_file}`")

        logging.info(
            f"Benchmarking inference rate for synchronous inference on {frames} frames "
            f"with {repeats} repeats"
        )
        with logging_redirect_tqdm(tqdm_class=tqdm), open(
            results_file, "w", newline=""
        ) as csvfile:
            results: List[Dict[str, str]] = []
            for index, deployment_folder in enumerate(
                tqdm(self._deployment_folders, desc="Benchmarking")
            ):
                success = True
                deployment = Deployment.from_folder(deployment_folder)
                try:
                    with suppress_log_output():
                        deployment.load_inference_models(device=target_device)
                except Exception as e:
                    success = False
                    logging.info(
                        f"Failed to load inference models for deployment at path: "
                        f"`{deployment_folder}`, with error: {e}. Marking benchmark "
                        f"run for the deployment as failed"
                    )

                if success:
                    try:
                        # Warm-up for the model
                        deployment.infer(benchmark_frames[0])

                        # Estimate time to completion
                        t_single_start = time.time()
                        deployment.infer(benchmark_frames[0])
                        single_inf_time = time.time() - t_single_start
                        logging.info(
                            f"Inference model(s) for deployment `{deployment_folder}` "
                            f"loaded. Starting benchmark run. Estimated time required: "
                            f"{repeats*frames*single_inf_time:.0f} seconds"
                        )
                    except Exception as e:
                        success = False
                        logging.info(
                            f"Failed to run inference on frame number 0. Marking "
                            f"benchmark run for deployment `{deployment_folder}` as "
                            f"failed. Inference failed with error: `{e}`"
                        )
                    t_start = time.time()
                    for run in range(repeats):
                        for frame in benchmark_frames:
                            try:
                                deployment.infer(frame)
                            except Exception as e:
                                success = False
                                logging.info(
                                    f"Failed to run inference on frame number "
                                    f"{benchmark_frames.index(frame)}. Marking "
                                    f"benchmark run for deployment "
                                    f"`{deployment_folder}` as failed. Inference "
                                    f"failed with error: `{e}`"
                                )
                    t_elapsed = time.time() - t_start
                    fps = frames * repeats / t_elapsed
                else:
                    fps = 0

                model_scores = []
                for om in deployment.models:
                    if isinstance(om.performance, Performance):
                        score = om.performance.score
                    elif isinstance(om.performance, dict):
                        score = om.performance.get("score", -1)
                    else:
                        score = -1
                    model_scores.append(score)

                # Update result list
                result_row: Dict[str, str] = {}
                result_row["name"] = f"Deployment {index}"
                result_row["project_name"] = self.project.name
                result_row["target_device"] = target_device
                result_row["task 1"] = self.project.get_trainable_tasks()[0].title
                result_row["model 1"] = deployment.models[0].name
                result_row["model 1 score"] = f"{model_scores[0]:.2f}"
                if not self._is_single_task:
                    result_row["task 2"] = self.project.get_trainable_tasks()[1].title
                    result_row["model 2"] = deployment.models[1].name
                    result_row["model 2 score"] = f"{model_scores[1]:.2f}"
                result_row["success"] = str(int(success))
                result_row["fps"] = f"{fps:.2f}"
                result_row["total frames"] = f"{frames * repeats}"
                result_row["source"] = deployment_folder
                result_row.update(get_system_info(device=target_device))
                results.append(result_row)

                # Write results to file
                if index == 0:
                    fieldnames = list(result_row.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(result_row)

        return results
