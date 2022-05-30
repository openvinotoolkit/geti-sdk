from enum import Enum


class AnnotationState(Enum):
    TO_REVISIT = 'to_revisit'
    ANNOTATED = 'annotated'
    PARTIALLY_ANNOTATED = 'partially_annotated'
    NONE = 'none'

    def __str__(self):
        """
        Returns the string representation of the AnnotationState instance
        :return:
        """
        return self.value
