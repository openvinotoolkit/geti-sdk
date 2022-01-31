import attr

from sc_api_tools.data_models import Annotation
from sc_api_tools.data_models.shapes import Rectangle


@attr.s(auto_attribs=True)
class ROI(Annotation):
    """
    This class represents a region of interest for a given image. ROIs are generated for
    intermediate tasks in the pipeline of a project, if those tasks produce local
    labels (for instance a detection or segmentation task).
    """
    shape: Rectangle = attr.ib(kw_only=True)

    @classmethod
    def from_annotation(cls, annotation: Annotation) -> 'ROI':
        """
        Converts an Annotation instance into an ROI

        :param annotation: Annotation to convert to region of interest
        :return: ROI containing the annotation
        """
        return ROI(labels=annotation.labels, shape=annotation.shape.to_roi())

    def to_absolute_coordinates(self, parent_roi: 'ROI') -> 'ROI':
        """
        Converts the ROI to an ROI in absolute coordinates, given it's parent ROI.

        :param parent_roi: Parent ROI containing the roi instance
        :return: ROI converted to the coordinate system of the parent ROI
        """
        return ROI(
            labels=self.labels,
            shape=self.shape.to_absolute_coordinates(parent_roi.shape)
        )
