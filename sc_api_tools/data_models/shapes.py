import abc
from typing import List, Tuple, TypeVar, Union

import attr

import numpy as np

from ote_sdk.entities.shapes.shape import (
    ShapeType as OteShapeType
)
from ote_sdk.entities.shapes.rectangle import Rectangle as OteRectangle
from ote_sdk.entities.shapes.ellipse import Ellipse as OteEllipse
from ote_sdk.entities.shapes.polygon import Polygon as OtePolygon

from sc_api_tools.data_models.enums import ShapeType
from sc_api_tools.data_models.utils import str_to_shape_type

OteShapeTypeVar = TypeVar('OteShapeTypeVar', OtePolygon, OteEllipse, OtePolygon)


@attr.s(auto_attribs=True)
class Shape:
    """
    Class representing a shape in SC

    :var type: Type of the shape
    """
    type: str = attr.ib(converter=str_to_shape_type)

    @abc.abstractmethod
    def to_roi(self) -> 'Rectangle':
        """
        Returns the bounding box containing the shape, as an instance of the Rectangle
        class

        :return: Rectangle representing the bounding box for the shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_absolute_coordinates(self, parent_roi: 'Rectangle') -> 'Shape':
        """
        Converts the Shape to absolute coordinates, given the rectangle
        representing it's parent region of interest.

        :param parent_roi: Region of interest containing the shape
        :return: Shape converted to the coordinate system of it's parent roi
        """
        raise NotImplementedError

    @classmethod
    def from_ote(
            cls, ote_shape: OteShapeTypeVar
    ) -> Union['Rectangle', 'Ellipse', 'Polygon']:
        """
        Creates a Shape entity from a corresponding shape in the OTE SDK.

        :param ote_shape: OTE SDK shape to convert from
        :return: Shape entity created from the ote_shape
        """
        shape_mapping = {
            OteShapeType.RECTANGLE: Rectangle,
            OteShapeType.ELLIPSE: Ellipse,
            OteShapeType.POLYGON: Polygon
        }
        return shape_mapping[ote_shape.type].from_ote(ote_shape)


@attr.s(auto_attribs=True)
class Rectangle(Shape):
    """
    Class representing a Rectangle in SC, as used in the /annotations REST endpoints

    NOTE: All coordinates and dimensions are relative to the full image

    :var x: X coordinate of the left side of the rectangle
    :var y: Y coordinate of the top of the rectangle
    :var width: Width of the rectangle
    :var height: Height of the rectangle
    """
    x: float
    y: float
    width: float
    height: float
    type: str = attr.ib(
        converter=str_to_shape_type, default=ShapeType.RECTANGLE, kw_only=True
    )

    def to_pixel_coordinates(
            self, image_width: int, image_height: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Gets the x, y coordinates of the edges of the rectangle, in number of pixels,
        if the rectangle would be applied to an image with dimensions `image_heigth x
        image_width` pixels

        :param image_width: Width of the image for which to get the pixel coordinates
            of the rectangle
        :param image_height: Height of the image for which to get the pixel coordinates
            of the rectangle
        :return: Tuple containing two pairs of integers representing:
            (x_min, x_max), (y_min, y_max)
        """
        return (
                (int(self.x*image_width), int((self.x+self.width)*image_width)),
                (int(self.y*image_height), int((self.y+self.height)*image_height))
        )

    @property
    def is_full_box(self) -> bool:
        """
        Returns True if this Rectangle represents a box containing the full image

        :return: True if the Rectangle encompasses the full image, False otherwise
        """
        return (
            self.x == 0.0
            and self.y == 0.0
            and self.width == 1.0
            and self.height == 1.0
        )

    def to_roi(self) -> 'Rectangle':
        """
        Returns the bounding box containing the shape, as an instance of the Rectangle
        class

        :return: Rectangle representing the bounding box for the shape
        """
        return self

    def to_absolute_coordinates(self, parent_roi: 'Rectangle') -> 'Rectangle':
        """
        Converts the Rectangle to absolute coordinates, given the rectangle
        representing it's parent region of interest.

        :param parent_roi: Region of interest containing the rectangle
        :return: Rectangle converted to the coordinate system of it's parent
        """
        x_min = parent_roi.x + self.x*parent_roi.width
        y_min = parent_roi.y + self.y*parent_roi.height
        width = parent_roi.width * self.width
        height = parent_roi.height * self.height
        return Rectangle(
                x=x_min, y=y_min, width=width, height=height
            )

    @classmethod
    def from_ote(cls, ote_shape: OteRectangle) -> 'Rectangle':
        """
        Creates a :py:class`~sc_api_tools.data_models.shapes.Rectangle` from
        the OTE SDK Rectangle entity passed.

        :param ote_shape: OTE SDK Rectangle entity to convert from
        :return: Rectangle instance created according to the ote_shape
        """
        return cls(
            x=ote_shape.x1,
            y=ote_shape.y1,
            width=ote_shape.width,
            height=ote_shape.height
        )


@attr.s(auto_attribs=True)
class Ellipse(Shape):
    """
    Class representing an Ellipse in SC, as used in the /annotations REST endpoints

    NOTE: All coordinates and dimensions are relative to the full image

    :var x: Lowest x coordinate of the ellipse
    :var y: Lowest y coordinate of the ellipse
    :var width: Width of the ellipse
    :var height: Height of the ellipse
    """
    x: float
    y: float
    width: float
    height: float
    type: str = attr.ib(
        converter=str_to_shape_type, default=ShapeType.ELLIPSE, kw_only=True
    )

    def to_roi(self) -> 'Rectangle':
        """
        Returns the bounding box containing the Ellipse, as an instance of the Rectangle
        class

        :return: Rectangle representing the bounding box for the ellipse
        """
        return Rectangle(x=self.x, y=self.y, width=self.width, height=self.height)

    def to_absolute_coordinates(self, parent_roi: Rectangle) -> 'Ellipse':
        """
        Converts the Ellipse to absolute coordinates, given the rectangle
        representing it's parent region of interest.

        :param parent_roi: Region of interest containing the ellipse
        :return: Ellipse converted to the coordinate system of it's parent
        """
        x_min = parent_roi.x + self.x*parent_roi.width
        y_min = parent_roi.y + self.y*parent_roi.height
        width = parent_roi.width * self.width
        height = parent_roi.height * self.height
        return Ellipse(
                x=x_min, y=y_min, width=width, height=height
            )

    @classmethod
    def from_ote(cls, ote_shape: OteEllipse) -> 'Ellipse':
        """
        Creates a :py:class`~sc_api_tools.data_models.shapes.Ellipse` from
        the OTE SDK Ellipse entity passed.

        :param ote_shape: OTE SDK Ellipse entity to convert from
        :return: Ellipse instance created according to the ote_shape
        """
        return cls(
            x=ote_shape.x1,
            y=ote_shape.y1,
            width=ote_shape.width,
            height=ote_shape.height
        )


@attr.s(auto_attribs=True)
class Point:
    """
    Class representing a point on a 2D coordinate system. Used to define Polygons in SC

    NOTE: All coordinates are defined relative to the full image size

    :var x: X coordinate of the point
    :var y: Y coordinate of the point
    """
    x: float
    y: float


@attr.s(auto_attribs=True)
class Polygon(Shape):
    """
    Class representing a polygon in SC, as used in the /annotations REST endpoints

    :var points: List of Points that make up the polygon
    """
    points: List[Point]
    type: str = attr.ib(
        converter=str_to_shape_type, default=ShapeType.POLYGON, kw_only=True
    )

    def points_as_contour(self, image_width: int, image_height: int) -> np.ndarray:
        """
        Returns the list of points for this Polygon as a numpy array representing
        contour points that can be plotted by openCV's drawContours function

        :param image_width: width of the image to which the shape should be applied
        :param image_height: height of the image to which the shape should be applied
        :return: Numpy array containing the contour
        """
        return np.array(
            [
                np.array((int(point.x*image_width), int(point.y*image_height)))
                for point in self.points
            ]
        )

    def to_roi(self) -> 'Rectangle':
        """
        Returns the bounding box containing the Polygon, as an instance of the Rectangle
        class

        :return: Rectangle representing the bounding box for the polygon
        """
        points_array = np.array([(point.x, point.y) for point in self.points])
        min_xy = points_array.min(axis=0)
        max_xy = points_array.max(axis=0)

        return Rectangle(
            x=min_xy[0],
            y=min_xy[1],
            width=max_xy[0] - min_xy[0],
            height=max_xy[1] - min_xy[1]
        )

    def to_absolute_coordinates(self, parent_roi: Rectangle) -> 'Polygon':
        """
        Converts the Polygon to absolute coordinates, given the rectangle
        representing it's parent region of interest.

        :param parent_roi: Region of interest containing the polygon
        :return: Polygon converted to the coordinate system of it's parent ROI
        """
        absolute_points = [
            Point(
                x=parent_roi.x + point.x*parent_roi.width,
                y=parent_roi.y + point.y*parent_roi.height
            ) for point in self.points
        ]
        return Polygon(points=absolute_points)

    @classmethod
    def from_ote(cls, ote_shape: OtePolygon) -> 'Polygon':
        """
        Creates a :py:class`~sc_api_tools.data_models.shapes.Polygon` from
        the OTE SDK Polygon entity passed.

        :param ote_shape: OTE SDK Polygon entity to convert from
        :return: Polygon instance created according to the ote_shape
        """
        points = [Point(x=ote_point.x, y=ote_point.y) for ote_point in ote_shape.points]
        return cls(
            points=points
        )
