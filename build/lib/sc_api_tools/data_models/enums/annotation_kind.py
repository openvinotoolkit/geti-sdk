from enum import Enum


class AnnotationKind(Enum):
    ANNOTATION = 'annotation'
    PREDICTION = 'prediction'

    def __str__(self):
        """
        Returns the string representation of the AnnotationKind instance
        :return:
        """
        return self.value
