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
from geti_sdk.data_models.enums import ShapeType
from geti_sdk.rest_converters.annotation_rest_converter import (
    NormalizedAnnotationRESTConverter,
)


class TestNormalizedAnnotationRESTConverter:
    def test_normalized_annotation_from_dict(self, fxt_normalized_annotation_dict):
        # Arrange
        img_width = 1000
        img_heigth = 2000
        norm_x = fxt_normalized_annotation_dict["shape"]["x"]
        norm_y = fxt_normalized_annotation_dict["shape"]["y"]
        norm_width = fxt_normalized_annotation_dict["shape"]["width"]
        norm_height = fxt_normalized_annotation_dict["shape"]["height"]

        # Act
        annotation = NormalizedAnnotationRESTConverter.normalized_annotation_from_dict(
            fxt_normalized_annotation_dict,
            image_width=img_width,
            image_height=img_heigth,
        )

        # Assert
        assert annotation.shape.type == ShapeType.RECTANGLE
        assert annotation.shape.x == int(norm_x * img_width)
        assert annotation.shape.y == int(norm_y * img_heigth)
        assert annotation.shape.width == int(norm_width * img_width)
        assert annotation.shape.height == int(norm_height * img_heigth)
