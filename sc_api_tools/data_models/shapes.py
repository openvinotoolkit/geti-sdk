from typing import List, Tuple

import attr

import numpy as np

from sc_api_tools.data_models.utils import str_to_shape_type


@attr.s(auto_attribs=True)
class Shape:
    """
    Class representing a shape in SC

    :var type: Type of the shape
    """
    type: str = attr.ib(converter=str_to_shape_type)


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

    def points_as_contour(self, image_width: int, image_height: int) -> np.ndarray:
        """
        Returns the list of points for this Polygon as a numpy array representing
        contour points that can be plotted by openCV's drawContours function

        :param image_width: width of the image to which the shape should be applied
        :param image_height: heigth of the image to which the shape should be applied
        :return: Numpy array containing the contour
        """
        return np.array(
            [
                np.array((int(point.x*image_width), int(point.y*image_height)))
                for point in self.points
            ]
        )
