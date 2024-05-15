# INTEL CONFIDENTIAL
#
# Copyright (C) 2024 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

import json
import logging
from typing import Any, Dict, List

import numpy as np
from model_api.models.utils import Detection

from geti_sdk.data_models.model import Model

# from sc_sdk.entities.model import Model

logger = logging.getLogger(__name__)


def detection2array(detections: List[Detection]) -> np.ndarray:
    """
    Convert list of OpenVINO Detection to a numpy array.

    :param detections: list of OpenVINO Detection containing [score, id, xmin, ymin, xmax, ymax]
    :return: numpy array with [label, confidence, x1, y1, x2, y2]
    """
    scores = np.empty((0, 1), dtype=np.float32)
    labels = np.empty((0, 1), dtype=np.uint32)
    boxes = np.empty((0, 4), dtype=np.float32)
    for det in detections:
        if (det.xmax - det.xmin) * (det.ymax - det.ymin) < 1.0:
            continue
        scores = np.append(scores, [[det.score]], axis=0)
        labels = np.append(labels, [[det.id]], axis=0)
        boxes = np.append(
            boxes,
            [[float(det.xmin), float(det.ymin), float(det.xmax), float(det.ymax)]],
            axis=0,
        )
    return np.concatenate((labels, scores, boxes), -1)


def get_detection_inferencer_configuration(model: Model) -> dict:
    """
    Get detection configuration from the model.

    :param model: (Geti) Model to get the detection configuration from
    :return: dict representing the detection configuration
    """
    config = json.loads(model.get_data("config.json"))
    _flatten_config_values(config)

    configuration = {}
    if config["postprocessing"].get("result_based_confidence_threshold", False):
        configuration["confidence_threshold"] = float(
            np.frombuffer(model.get_data("confidence_threshold"), dtype=np.float32)[0]
        )
    configuration["use_ellipse_shapes"] = config["postprocessing"].get(
        "use_ellipse_shapes", False
    )

    logger.info(f"Detection inferencer configuration: {configuration}")
    return configuration


def _flatten_config_values(config: Dict[str, Any]) -> None:
    """
    Extract the "value" field from any nested config.

    Flattening the structure of the config dictionary. The original config dictionary is modified in-place.

    :param config: config dictionary
    """
    for key, value in config.items():
        if isinstance(value, dict):
            if "value" in value:
                config[key] = value["value"]
            else:
                _flatten_config_values(value)
