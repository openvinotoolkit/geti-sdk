# INTEL CONFIDENTIAL
#
# Copyright (C) 2023 Intel Corporation
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

"""
The package contains classes for inference results post-processing and conversion to internal Prediction entities.

The package contains the following classes:
    - `AnomalyToPredictionConverter` - class for converting anomaly classification / segmentation / detection results to internal Prediction entities
    - `ClassificationToPredictionConverter` - class for converting classification results to internal Prediction entities
    - `DetectionToPredictionConverter` - class for converting detection results to internal Prediction entities
    - `MaskToAnnotationConverter` - class for converting rotated detection results to internal Prediction entities
    - `RotatedRectToPredictionConverter` - class for converting rotated detection results to internal Prediction entities
    - `SegmentationToPredictionConverter` - class for converting segmentation results to internal Prediction entities

    - `ConverterFactory` - factory class for creating the appropriate converter based on the domain of the inference results
"""

from .results_to_prediction_converter import ConverterFactory

__all__ = ["ConverterFactory"]
