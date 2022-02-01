from typing import List, Optional

import attr

import numpy as np

from sc_api_tools.data_models import Prediction, Label, Annotation

from .region_of_interest import ROI


@attr.s(auto_attribs=True)
class IntermediateInferenceResult:
    """
    This class represents the inference results for intermediate tasks in the pipeline
    """
    prediction: Prediction
    image: np.ndarray
    rois: Optional[List[ROI]] = None

    @property
    def image_width(self) -> int:
        """
        Returns the width of the image to which the InferenceResult applies

        :return: Integer representing the width of the image, in pixels
        """
        return self.image.shape[1]

    @property
    def image_height(self) -> int:
        """
        Returns the height of the image to which the InferenceResult applies

        :return: Integer representing the height of the image, in pixels
        """
        return self.image.shape[0]

    def filter_rois(self, label: Optional[Label] = None) -> List[ROI]:
        """
        Filters the ROIs for the inference results based on an input label

        :param label: Label to retrieve the ROIs for. If left as None, all the ROIs
            belonging to the inference result are returned
        :return: List of ROIs containing an object with the specified label
        """
        if label is None:
            return self.rois
        return [roi for roi in self.rois if label.name in roi.label_names]

    def generate_views(self, rois: Optional[List[ROI]] = None) -> List[np.ndarray]:
        """
        Generates a list of image views holding the pixel data for the ROIs produced
        by the last local-label task in the pipeline

        :param rois: Optional list of ROIs to return the views for. If left as None,
            views for all ROIs are returned.
        :return: List of numpy arrays containing the pixel data for the ROI's in the
            list of ROI's associated with this inference result
        """
        if self.rois is None:
            return [self.image]

        if rois is not None:
            denormalized_rois = [
                roi.shape.to_pixel_coordinates(
                    image_width=self.image_width, image_height=self.image_height
                ) for roi in self.rois if roi in rois
            ]
        else:
            denormalized_rois = [
                roi.shape.to_pixel_coordinates(
                    image_width=self.image_width, image_height=self.image_height
                ) for roi in self.rois
            ]

        if len(self.image.shape) == 3:
            return [
                self.image[roi[1][0]:roi[1][1], roi[0][0]:roi[0][1], :]
                for roi in denormalized_rois
            ]
        elif len(self.image.shape) == 2:
            return [
                self.image[roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]]
                for roi in denormalized_rois
            ]
        else:
            raise ValueError(
                f"Unexpected image shape: {self.image.shape}. Unable to generate image "
                f"views"
            )

    def append_annotation(self, annotation: Annotation, roi: ROI):
        """
        Appends an Annotation instance to the prediction results, taking into account
        the ROI for which the annotation was predicted

        This method can be used to add annotations produced by a downstream local task
        to the prediction results

        :param annotation: Annotation to append to the inference results
        :param roi: ROI in which the prediction was made
        """
        absolute_shape = annotation.shape.to_absolute_coordinates(parent_roi=roi.shape)
        self.prediction.append(
            Annotation(labels=annotation.labels, shape=absolute_shape)
        )

    def extend_annotations(self, annotations: List[Annotation], roi: ROI):
        """
        Extends the list of annotations for the current prediction results, taking
        into account the ROI for which the annotation was predicted

        This method can be used to add labels produced by a global downstream task to
        the ROI output of it's upstream local task

        :param annotations: List of annotations holding the labels to append
        :param roi: ROI for which the annotations are predicted
        """
        for annotation in annotations:
            annotation.shape = roi.original_shape
        self.prediction.extend(annotations)
