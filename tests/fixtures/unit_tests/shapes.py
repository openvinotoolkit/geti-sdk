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

import pytest

from geti_sdk.data_models.shapes import (
    Ellipse,
    Keypoint,
    Point,
    Polygon,
    Rectangle,
    RotatedRectangle,
)


@pytest.fixture()
def fxt_rectangle() -> Rectangle:
    yield Rectangle(x=50, y=200, width=900, height=1600)


@pytest.fixture()
def fxt_rectangle_roi() -> Rectangle:
    yield Rectangle(x=100, y=200, width=1800, height=3200)


@pytest.fixture()
def fxt_ellipse() -> Ellipse:
    yield Ellipse(x=50, y=200, width=900, height=1600)


@pytest.fixture()
def fxt_triangle() -> Polygon:
    yield Polygon(
        points=[
            Point(x=10, y=20),
            Point(x=30, y=50),
            Point(x=50, y=20),
        ]
    )


@pytest.fixture()
def fxt_rotated_rectangle() -> RotatedRectangle:
    yield RotatedRectangle(x=200, y=200, width=50, height=100, angle=45)


@pytest.fixture()
def fxt_rotated_rectangle_as_polygon() -> Polygon:
    yield Polygon(
        points=[
            Point(x=146, y=217),
            Point(x=217, y=146),
            Point(x=253, y=182),
            Point(x=182, y=253),
        ]
    )


@pytest.fixture()
def fxt_keypoint() -> Keypoint:
    yield Keypoint(x=100, y=199, is_visible=True)
