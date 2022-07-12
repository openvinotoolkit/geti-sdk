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

import abc
import math
from typing import Any, Dict, List, Optional, TypeVar, Union

import attr
import cv2
import numpy as np
from ote_sdk.entities.shapes.ellipse import Ellipse as OteEllipse
from ote_sdk.entities.shapes.polygon import Polygon as OtePolygon
from ote_sdk.entities.shapes.rectangle import Rectangle as OteRectangle
from ote_sdk.entities.shapes.shape import ShapeType as OteShapeType

from sc_api_tools.data_models.enums import ShapeType
from sc_api_tools.data_models.utils import round_to_n_digits, str_to_shape_type

# OteShapeTypeVar is a type variable that represents the possible different shapes
# that can be converted from the OTE SDK
OteShapeTypeVar = TypeVar("OteShapeTypeVar", OtePolygon, OteEllipse, OtePolygon)

# N_DIGITS_TO_ROUND_TO determines how pixel coordinates will be rounded when they are
# passed from the SC REST API. SC itself rounds some coordinates to 4 digits, but not
# all. Here we round all coordinates for internal consistency
N_DIGITS_TO_ROUND_TO = 4
coordinate_converter = round_to_n_digits(N_DIGITS_TO_ROUND_TO)


@attr.s(auto_attribs=True)
class Shape:
    """
    Representation of a shape in SC.

    :var type: Type of the shape
    """

    type: str = attr.ib(converter=str_to_shape_type)

    @abc.abstractmethod
    def to_roi(self) -> "Rectangle":
        """
        Return the bounding box containing the shape, as an instance of the Rectangle
        class.

        :return: Rectangle representing the bounding box for the shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_absolute_coordinates(self, parent_roi: "Rectangle") -> "Shape":
        """
        Convert the Shape to absolute coordinates, given the rectangle
        representing it's parent region of interest.

        :param parent_roi: Region of interest containing the shape
        :return: Shape converted to the coordinate system of it's parent roi
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_normalized_coordinates(
        self, image_width: int, image_height: int
    ) -> Dict[str, Any]:
        """
        Convert the Shape to a normalized coordinate system, such that all
        coordinates are represented as floats in the interval [0, 1].

        :param image_width: Width (in pixels) of the image or region with respect to
            which the shape coordinates should be normalized.
        :param image_height: Height (in pixels) of the image or region with respect to
            which the shape coordinates should be normalized
        :return: Dictionary representing the Shape object with it's coordinates
            normalized with respect to image_width and image_height
        """
        raise NotImplementedError

    @classmethod
    def from_ote(
        cls, ote_shape: OteShapeTypeVar, image_width: int, image_height: int
    ) -> Union["Rectangle", "Ellipse", "Polygon", "RotatedRectangle"]:
        """
        Create a Shape entity from a corresponding shape in the OTE SDK.

        :param ote_shape: OTE SDK shape to convert from
        :param image_width: Width of the image to which the shape applies (in pixels)
        :param image_height: Heigth of the image to which the shape applies (in pixels)
        :return: Shape entity created from the ote_shape
        """
        shape_mapping = {
            OteShapeType.RECTANGLE: Rectangle,
            OteShapeType.ELLIPSE: Ellipse,
            OteShapeType.POLYGON: Polygon,
        }
        return shape_mapping[ote_shape.type].from_ote(
            ote_shape, image_width=image_width, image_height=image_height
        )

    @property
    @abc.abstractmethod
    def area(self) -> float:
        """
        Return the area of the shape, given in number of pixels.

        :return: Area of the shape, in pixels
        """
        raise NotImplementedError


@attr.s(auto_attribs=True)
class Rectangle(Shape):
    """
    Representation of a Rectangle in SC, as used in the /annotations REST endpoints.

    NOTE: All coordinates and dimensions are given in pixels

    :var x: X coordinate of the left side of the rectangle
    :var y: Y coordinate of the top of the rectangle
    :var width: Width of the rectangle
    :var height: Height of the rectangle
    """

    x: float = attr.ib(converter=coordinate_converter)
    y: float = attr.ib(converter=coordinate_converter)
    width: float = attr.ib(converter=coordinate_converter)
    height: float = attr.ib(converter=coordinate_converter)
    type: str = attr.ib(
        converter=str_to_shape_type, default=ShapeType.RECTANGLE, kw_only=True
    )

    def to_normalized_coordinates(
        self, image_width: int, image_height: int
    ) -> Dict[str, float]:
        """
        Get the normalized coordinates of the rectangle, with respect to the image
        with dimensions `image_width` x `image_height`.

        :param image_width: Width of the image to which the coordinates should be
            normalized
        :param image_height: Height of the image to which the coordinates should be
            normalized
        :return: Dictionary containing the rectangle, represented in normalized
            coordinates
        """
        return dict(
            x=self.x / image_width,
            y=self.y / image_height,
            width=self.width / image_width,
            height=self.height / image_height,
            type=str(self.type),
        )

    def is_full_box(self, image_width: int, image_height: int) -> bool:
        """
        Return True if this Rectangle represents a box containing the full image.

        :param image_width: Width of the image to check for
        :param image_height: Height of the image to check for
        :return: True if the Rectangle encompasses the full image, False otherwise
        """
        return (
            self.x == 0
            and self.y == 0
            and int(self.width) == image_width
            and int(self.height) == image_height
        )

    def to_roi(self) -> "Rectangle":
        """
        Return the bounding box containing the shape, as an instance of the Rectangle
        class.

        :return: Rectangle representing the bounding box for the shape
        """
        return self

    def to_absolute_coordinates(self, parent_roi: "Rectangle") -> "Rectangle":
        """
        Convert the Rectangle to absolute coordinates, given the rectangle
        representing it's parent region of interest.

        :param parent_roi: Region of interest containing the rectangle
        :return: Rectangle converted to the coordinate system of it's parent
        """
        x_min = parent_roi.x + self.x
        y_min = parent_roi.y + self.y
        return Rectangle(x=x_min, y=y_min, width=self.width, height=self.height)

    @classmethod
    def from_ote(
        cls, ote_shape: OteRectangle, image_width: int, image_height: int
    ) -> "Rectangle":
        """
        Create a :py:class`~sc_api_tools.data_models.shapes.Rectangle` from
        the OTE SDK Rectangle entity passed.

        :param ote_shape: OTE SDK Rectangle entity to convert from
        :param image_width: Width of the image to which the rectangle applies (in pixels)
        :param image_height: Heigth of the image to which the rectangle applies (in pixels)
        :return: Rectangle instance created according to the ote_shape
        """
        return cls(
            x=ote_shape.x1 * image_width,
            y=ote_shape.y1 * image_height,
            width=ote_shape.width * image_width,
            height=ote_shape.height * image_height,
        )

    @property
    def area(self) -> float:
        """
        Return the area of the Rectangle, in pixels.

        :return:
        """
        return self.width * self.height


@attr.s(auto_attribs=True)
class Ellipse(Shape):
    """
    Representation of an Ellipse in SC, as used in the /annotations REST endpoints.

    NOTE: All coordinates and dimensions are given in pixels

    :var x: Lowest x coordinate of the ellipse
    :var y: Lowest y coordinate of the ellipse
    :var width: Width of the ellipse
    :var height: Height of the ellipse
    """

    x: float = attr.ib(converter=coordinate_converter)
    y: float = attr.ib(converter=coordinate_converter)
    width: float = attr.ib(converter=coordinate_converter)
    height: float = attr.ib(converter=coordinate_converter)
    type: str = attr.ib(
        converter=str_to_shape_type, default=ShapeType.ELLIPSE, kw_only=True
    )

    def to_roi(self) -> "Rectangle":
        """
        Return the bounding box containing the Ellipse, as an instance of the Rectangle
        class.

        :return: Rectangle representing the bounding box for the ellipse
        """
        return Rectangle(x=self.x, y=self.y, width=self.width, height=self.height)

    def to_absolute_coordinates(self, parent_roi: Rectangle) -> "Ellipse":
        """
        Convert the Ellipse to absolute coordinates, given the rectangle
        representing it's parent region of interest.

        :param parent_roi: Region of interest containing the ellipse
        :return: Ellipse converted to the coordinate system of it's parent
        """
        x_min = parent_roi.x + self.x
        y_min = parent_roi.y + self.y
        return Ellipse(x=x_min, y=y_min, width=self.width, height=self.height)

    def to_normalized_coordinates(
        self, image_width: int, image_height: int
    ) -> Dict[str, float]:
        """
        Get the normalized coordinates of the ellipse, with respect to the image
        with dimensions `image_width` x `image_height`.

        :param image_width: Width of the image to which the coordinates should be
            normalized
        :param image_height: Height of the image to which the coordinates should be
            normalized
        :return: Dictionary containing the ellipse, represented in normalized
            coordinates
        """
        return dict(
            x=self.x / image_width,
            y=self.y / image_height,
            width=self.width / image_width,
            heigth=self.height / image_height,
            type=str(self.type),
        )

    @classmethod
    def from_ote(
        cls, ote_shape: OteEllipse, image_width: int, image_height: int
    ) -> "Ellipse":
        """
        Create a :py:class`~sc_api_tools.data_models.shapes.Ellipse` from
        the OTE SDK Ellipse entity passed.

        :param ote_shape: OTE SDK Ellipse entity to convert from
        :param image_width: Width of the image to which the ellipse applies (in pixels)
        :param image_height: Heigth of the image to which the ellipse applies (in pixels)
        :return: Ellipse instance created according to the ote_shape
        """
        return cls(
            x=ote_shape.x1 * image_width,
            y=ote_shape.y1 * image_height,
            width=ote_shape.width * image_width,
            height=ote_shape.height * image_height,
        )

    @property
    def area(self) -> float:
        """
        Return the area of the Ellipse, in pixels.

        :return:
        """
        return self.width * self.height * math.pi


@attr.s(auto_attribs=True)
class Point:
    """
    Representation of a point on a 2D coordinate system. Used to define Polygons in SC.

    NOTE: All coordinates are defined in pixels

    :var x: X coordinate of the point
    :var y: Y coordinate of the point
    """

    x: float = attr.ib(converter=coordinate_converter)
    y: float = attr.ib(converter=coordinate_converter)


@attr.s(auto_attribs=True)
class Polygon(Shape):
    """
    Representation of a polygon in SC, as used in the /annotations REST endpoints.

    :var points: List of Points that make up the polygon
    """

    points: List[Point]
    type: str = attr.ib(
        converter=str_to_shape_type, default=ShapeType.POLYGON, kw_only=True
    )

    def __attrs_post_init__(self):
        """
        Initialize private attributes.
        """
        self._contour: Optional[np.ndarray] = None

    def points_as_contour(self) -> np.ndarray:
        """
        Return the list of points for this Polygon as a numpy array representing
        contour points that can be plotted by openCV's drawContours function.

        NOTE: the contour is cached, to avoid going over the list of points twice. If
        the polygon is modified the cache has to be cleared. It is recommended to
        create a new polygon instead

        :return: Numpy array containing the contour
        """
        if self._contour is None:
            self._contour = np.array(
                [(int(point.x), int(point.y)) for point in self.points]
            )
        return self._contour

    def to_roi(self) -> "Rectangle":
        """
        Return the bounding box containing the Polygon, as an instance of the Rectangle
        class.

        :return: Rectangle representing the bounding box for the polygon
        """
        points_array = np.array([(point.x, point.y) for point in self.points])
        min_xy = points_array.min(axis=0)
        max_xy = points_array.max(axis=0)

        return Rectangle(
            x=min_xy[0],
            y=min_xy[1],
            width=max_xy[0] - min_xy[0],
            height=max_xy[1] - min_xy[1],
        )

    def to_absolute_coordinates(self, parent_roi: Rectangle) -> "Polygon":
        """
        Convert the Polygon to absolute coordinates, given the rectangle
        representing it's parent region of interest.

        :param parent_roi: Region of interest containing the polygon
        :return: Polygon converted to the coordinate system of it's parent ROI
        """
        absolute_points = [
            Point(x=parent_roi.x + point.x, y=parent_roi.y + point.y)
            for point in self.points
        ]
        return Polygon(points=absolute_points)

    def to_normalized_coordinates(
        self, image_width: int, image_height: int
    ) -> Dict[str, Union[List[Dict[str, float]], str]]:
        """
        Get the normalized coordinates of the polygon, with respect to the image
        with dimensions `image_width` x `image_height`.

        :param image_width: Width of the image to which the coordinates should be
            normalized
        :param image_height: Height of the image to which the coordinates should be
            normalized
        :return: Dictionary containing the polygon, represented in normalized
            coordinates
        """
        normalized_points: List[Dict[str, float]] = []
        for point in self.points:
            normalized_points.append(
                {"x": point.x / image_width, "y": point.y / image_height}
            )
        return dict(points=normalized_points, type=str(self.type))

    @classmethod
    def from_ote(
        cls, ote_shape: OtePolygon, image_width: int, image_height: int
    ) -> "Polygon":
        """
        Create a :py:class`~sc_api_tools.data_models.shapes.Polygon` from
        the OTE SDK Polygon entity passed.

        :param ote_shape: OTE SDK Polygon entity to convert from
        :param image_width: Width of the image to which the polygon applies (in pixels)
        :param image_height: Heigth of the image to which the polygon applies (in pixels)
        :return: Polygon instance created according to the ote_shape
        """
        points = [
            Point(x=ote_point.x * image_width, y=ote_point.y * image_height)
            for ote_point in ote_shape.points
        ]
        return cls(points=points)

    @property
    def area(self) -> float:
        """
        Return the area of the Polygon, in pixels.

        :return: area of the polygon
        """
        return cv2.contourArea(self.points_as_contour())

    def fit_rotated_rectangle(self) -> "RotatedRectangle":
        """
        Fit a RotatedRectangle object around the Polygon, such that the area spanned
        by the rectangle is minimal.

        :return: RotatedRectangle object with minimal area, which encloses the Polygon
        """
        center, (width, height), angle = cv2.minAreaRect(self.points_as_contour())
        return RotatedRectangle(
            angle=angle, x=center[0], y=center[1], width=width, height=height
        )


@attr.s(auto_attribs=True)
class RotatedRectangle(Shape):
    """
    Representation of a RotatedRectangle in SC, as used in the /annotations REST
    endpoints.

    NOTE: All coordinates and dimensions are specified in pixels

    :var angle: angle, in degrees, under which the rectangle is defined.
    :var x: X coordinate of the center of the rectangle
    :var y: Y coordinate of the center of the rectangle
    :var width: Width of the rectangle
    :var height: Height of the rectangle
    """

    angle: float = attr.ib(converter=coordinate_converter)
    x: float = attr.ib(converter=coordinate_converter)
    y: float = attr.ib(converter=coordinate_converter)
    width: float = attr.ib(converter=coordinate_converter)
    height: float = attr.ib(converter=coordinate_converter)
    type: str = attr.ib(
        converter=str_to_shape_type, default=ShapeType.ROTATED_RECTANGLE, kw_only=True
    )

    @property
    def _angle_x_radian(self) -> float:
        """
        Return the angle to the x-axis, in radians.

        :return: angle to the x-axis in radians
        """
        return (2 * math.pi / 360) * self.angle

    @property
    def x_max(self) -> float:
        """
        Return the value of the maximal x-coordinate that the rotated rectangle touches.

        :return: Maximum x-coordinate for the rotated rectangle
        """
        return (
            self.x
            + 0.5 * self.width * math.cos(self._angle_x_radian)
            + 0.5 * self.height * math.sin(self._angle_x_radian)
        )

    @property
    def x_min(self) -> float:
        """
        Return the value of the minimal x-coordinate that the rotated rectangle touches.

        :return: Minimum x-coordinate for the rotated rectangle
        """
        return (
            self.x
            - 0.5 * self.width * math.cos(self._angle_x_radian)
            - 0.5 * self.height * math.sin(self._angle_x_radian)
        )

    @property
    def y_max(self) -> float:
        """
        Return the value of the maximal y-coordinate that the rotated rectangle touches.

        :return: Maximum y-coordinate for the rotated rectangle
        """
        return (
            self.y
            + 0.5 * self.width * math.sin(self._angle_x_radian)
            + 0.5 * self.height * math.cos(self._angle_x_radian)
        )

    @property
    def y_min(self):
        """
        Return the value of the minimal y-coordinate that the rotated rectangle touches.

        :return: Minimum y-coordinate for the rotated rectangle
        """
        return (
            self.y
            - 0.5 * self.width * math.sin(self._angle_x_radian)
            - 0.5 * self.height * math.cos(self._angle_x_radian)
        )

    @classmethod
    def from_polygon(cls, polygon: Polygon) -> "RotatedRectangle":
        """
        Create a :py:class`~sc_api_tools.data_models.shapes.RotatedRectangle` from
        the Polygon entity passed.

        NOTE: The Polygon MUST consist of 4 points, otherwise a ValueError is raised

        :param polygon: Polygon entity to convert from
        :return: Rectangle instance created according to the polygon object
        """
        if len(polygon.points) != 4:
            raise ValueError(
                f"Unable to convert polygon {polygon} to RotatedRectangle. A rotated "
                f"rectangle must have exactly 4 points."
            )

        x_coords = [point.x for point in polygon.points]
        y_coords = [point.y for point in polygon.points]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_center = x_min + 0.5 * (x_max - abs(x_min))
        y_center = y_min + 0.5 * (y_max - abs(y_min))

        if len(x_coords) > len(set(x_coords)) or len(y_coords) > len(set(y_coords)):
            # In this case there are points sharing the same x or y value, which means
            # that we have an angle of 0 degrees
            width = x_max - x_min
            height = y_max - y_min
            alpha = 0
        else:
            x_min_coord = Point(x=x_min, y=y_coords[x_coords.index(x_min)])
            x_max_coord = Point(x=x_max, y=y_coords[x_coords.index(x_max)])
            y_min_coord = Point(x=x_coords[y_coords.index(y_min)], y=y_min)

            # Calculate the angle to the x-axis, alpha
            perpendicular_side = x_max_coord.y - y_min_coord.y
            base_side = x_max_coord.x - y_min_coord.x
            hypotenuse = math.sqrt(perpendicular_side**2 + base_side**2)

            alpha = math.asin(perpendicular_side / hypotenuse)
            alpha = alpha * 360 / (2 * math.pi)

            width = hypotenuse

            # Calculate height
            perpendicular_side_2 = x_min_coord.y - y_min_coord.y
            base_side_2 = x_min_coord.x - y_min_coord.x
            height = math.sqrt(perpendicular_side_2**2 + base_side_2**2)

        return cls(x=x_center, y=y_center, width=width, height=height, angle=alpha)

    @classmethod
    def from_ote(
        cls, ote_shape: OtePolygon, image_width: int, image_height: int
    ) -> "RotatedRectangle":
        """
        Create a :py:class`~sc_api_tools.data_models.shapes.RotatedRectangle` from
        the OTE SDK Polygon entity passed.

        NOTE: The Polygon MUST consist of 4 points, otherwise a ValueError is raised

        :param ote_shape: OTE SDK Rectangle entity to convert from
        :param image_width: Width of the image to which the shape applies (in pixels)
        :param image_height: Heigth of the image to which the shape applies (in pixels)
        :return: RotatedRectangle instance created according to the ote_shape
        """
        polygon = Polygon.from_ote(
            ote_shape, image_width=image_width, image_height=image_height
        )
        return cls.from_polygon(polygon)

    def to_roi(self) -> Rectangle:
        """
        Return the bounding box containing the RotatedRectangle, as an instance of
        the Rectangle class.

        :return: Rectangle representing the bounding box for the rotated_rectangle
        """
        return Rectangle(
            x=self.x,
            y=self.y,
            width=self.x_max - self.x_min,
            height=self.y_max - self.y_min,
        )

    def to_absolute_coordinates(self, parent_roi: Rectangle) -> "RotatedRectangle":
        """
        Convert the RotatedRectangle to absolute coordinates, given the rectangle
        representing it's parent region of interest.

        :param parent_roi: Region of interest containing the rotated rectangle
        :return: RotatedRectangle converted to the coordinate system of it's parent ROI
        """
        x = parent_roi.x + self.x
        y = parent_roi.y + self.y

        width = self.width
        height = self.height
        return RotatedRectangle(x=x, y=y, width=width, height=height, angle=self.angle)

    def to_polygon(self) -> Polygon:
        """
        Convert the RotatedRectangle instance to a Polygon consisting of 4 points.

        :return: Polygon object corresponding to the RotatedRectangle instance
        """
        y_0 = (
            self.y
            - 0.5 * self.width * math.sin(self._angle_x_radian)
            + 0.5 * self.height * math.cos(self._angle_x_radian)
        )

        x_1 = (
            self.x
            - 0.5 * self.width * math.cos(self._angle_x_radian)
            + 0.5 * self.height * math.sin(self._angle_x_radian)
        )

        y_2 = (
            self.y
            + 0.5 * self.width * math.sin(self._angle_x_radian)
            - 0.5 * self.height * math.cos(self._angle_x_radian)
        )

        x_3 = (
            self.x
            + 0.5 * self.width * math.cos(self._angle_x_radian)
            - 0.5 * self.height * math.sin(self._angle_x_radian)
        )

        point0 = Point(x=self.x_min, y=y_0)
        point1 = Point(x=x_1, y=self.y_min)
        point2 = Point(x=self.x_max, y=y_2)
        point3 = Point(x=x_3, y=self.y_max)

        return Polygon(points=[point0, point1, point2, point3])

    def to_normalized_coordinates(
        self, image_width: int, image_height: int
    ) -> Dict[str, Union[float, str]]:
        """
        Get the normalized coordinates of the rotated rectangle, with respect to the
        image with dimensions `image_width` x `image_height`.

        :param image_width: Width of the image to which the coordinates should be
            normalized
        :param image_height: Height of the image to which the coordinates should be
            normalized
        :return: Dictionary containing the rotated rectangle, represented in normalized
            coordinates
        """
        return dict(
            angle=self.angle,
            x=self.x / image_width,
            y=self.y / image_height,
            width=self.width / image_width,
            height=self.height / image_height,
            type=str(self.type),
        )

    @property
    def area(self) -> float:
        """
        Return the area of the RotatedRectangle, in pixels

        :return: area enclosed by the rotated rectangle in pixels
        """
        return self.width * self.height
