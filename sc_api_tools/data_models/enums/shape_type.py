from enum import Enum


class ShapeType(Enum):
    """
    This Enum represents the types of shapes avalaible in SC
    """
    RECTANGLE = 'RECTANGLE'
    ELLIPSE = 'ELLIPSE'
    POLYGON = 'POLYGON'
    ROTATED_RECTANGLE = 'ROTATED_RECTANGLE'

    def __str__(self):
        """
        Returns the string representation of the ShapeType instance
        :return:
        """
        return self.value
