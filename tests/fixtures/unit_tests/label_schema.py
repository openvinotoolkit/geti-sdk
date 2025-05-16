# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
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

import pytest

from geti_sdk.data_models.containers import LabelList
from geti_sdk.data_models.enums.domain import Domain


@pytest.fixture
def fxt_label_list_factory(
    fxt_classification_labels,
    fxt_detection_labels,
    fxt_empty_detection_label,
    fxt_segmentation_labels,
    fxt_empty_segmentation_label,
    fxt_rotated_detection_labels,
    fxt_empty_rotated_detection_label,
    fxt_keypoint_labels,
    fxt_anomaly_labels_factory,
):
    domain_to_label_properties = {
        Domain.DETECTION: {
            "labels": fxt_detection_labels,
            "empty_label": fxt_empty_detection_label,
        },
        Domain.CLASSIFICATION: {
            "labels": fxt_classification_labels,
        },
        Domain.SEGMENTATION: {
            "labels": fxt_segmentation_labels,
            "empty_label": fxt_empty_segmentation_label,
        },
        Domain.ROTATED_DETECTION: {
            "labels": fxt_rotated_detection_labels,
            "empty_label": fxt_empty_rotated_detection_label,
        },
        Domain.ANOMALY_CLASSIFICATION: {
            "labels": fxt_anomaly_labels_factory(Domain.ANOMALY_CLASSIFICATION),
        },
        Domain.ANOMALY_SEGMENTATION: {
            "labels": fxt_anomaly_labels_factory(Domain.ANOMALY_SEGMENTATION),
        },
        Domain.ANOMALY_DETECTION: {
            "labels": fxt_anomaly_labels_factory(Domain.ANOMALY_DETECTION),
        },
        Domain.KEYPOINT_DETECTION: {
            "labels": fxt_keypoint_labels,
        },
    }

    def _label_list_factory(domain: Domain):
        labels = domain_to_label_properties[domain]["labels"]
        empty_label = domain_to_label_properties[domain].get("empty_label", None)
        if empty_label is not None:
            labels.append(empty_label)
        return LabelList(labels)

    yield _label_list_factory
