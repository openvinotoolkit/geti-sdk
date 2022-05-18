import copy
import warnings

from typing import List, Tuple, Sequence, Optional, cast

import numpy as np

import cv2

from sc_api_tools.data_models import Annotation, Label, ScoredLabel, Prediction
from sc_api_tools.data_models.shapes import Point, Polygon, Rectangle


Contour = List[Tuple[float, float]]


def get_subcontours(contour: Contour) -> List[Contour]:
    """
    Splits contour into subcontours that do not have self intersections.
    """

    ContourInternal = List[Optional[Tuple[float, float]]]

    def find_loops(points: ContourInternal) -> List[Sequence[int]]:
        """
        For each consecutive pair of equivalent rows in the input matrix
        returns their indices.
        """
        _, inverse, count = np.unique(
            points, axis=0, return_inverse=True, return_counts=True
        )
        duplicates = np.where(count > 1)[0]
        indices = []
        for x in duplicates:
            y = np.nonzero(inverse == x)[0]
            for i, _ in enumerate(y[:-1]):
                indices.append(y[i: i + 2])
        return indices

    base_contour = cast(ContourInternal, copy.copy(contour))

    # Make sure that contour is closed.
    if not np.array_equal(base_contour[0], base_contour[-1]):
        base_contour.append(base_contour[0])

    subcontours: List[Contour] = []
    loops = sorted(find_loops(base_contour), key=lambda x: x[0], reverse=True)
    for loop in loops:
        i, j = loop
        subcontour = base_contour[i:j]
        subcontour = list(x for x in subcontour if x is not None)
        subcontours.append(cast(Contour, subcontour))
        base_contour[i:j] = [None] * (j - i)

    subcontours = [i for i in subcontours if len(i) > 2]
    return subcontours


def convert_segmentation_output(
        model_output: np.ndarray,
        labels: List[Label],
        soft_prediction: Optional[np.ndarray] = None
) -> Prediction:
    """
    Creates polygons from the soft predictions.
    Background label will be ignored and not be converted to polygons.

    :param model_output: hard prediction containing the final label index per pixel.
        This is the prediction output produced after postprocessing from the model
    :param labels: List of labels in the segmentation task
    :param soft_prediction: Optional soft prediction with shape H x W x N_labels,
        where soft_prediction[:, :, 0] is the soft prediction for background.
        If soft_prediction is of H x W shape, it is assumed that this soft prediction
        will be applied for all labels. If left as None, the probability for each
        shape cannot be computed and will therefore be returned as 0
    :return: Prediction containing the output of the model
    """
    # pylint: disable=too-many-locals
    height, width = model_output.shape[:2]
    img_class = model_output.swapaxes(0, 1)

    # pylint: disable=too-many-nested-blocks
    annotations: List[Annotation] = []
    empty_label = next((label for label in labels if label.is_empty), None)
    compute_scores = soft_prediction is not None
    if model_output.sum() == 0:
        if empty_label is not None:
            score = (1 - cv2.mean(soft_prediction[:, :, 0])) if compute_scores else 0
            shape = Rectangle(x=0, y=0, width=1, height=1)
            labels = [ScoredLabel.from_label(label=empty_label, probability=score)]
            return Prediction(annotations=[Annotation(shape=shape, labels=labels)])

    if empty_label is not None:
        labels.pop(labels.index(empty_label))

    for label_index, label in enumerate(labels):
        # obtain current label soft prediction
        if compute_scores:
            if len(soft_prediction.shape) == 3:
                current_label_soft_prediction = soft_prediction[:, :, label_index + 1]
            else:
                current_label_soft_prediction = soft_prediction

        obj_group = img_class == label_index + 1
        label_index_map = (obj_group.T.astype(int) * 255).astype(np.uint8)

        # Contour retrieval mode CCOMP (Connected components) creates a two-level
        # hierarchy of contours
        contours, hierarchies = cv2.findContours(
            label_index_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        if hierarchies is not None:
            for contour, hierarchy in zip(contours, hierarchies[0]):
                if hierarchy[3] == -1:
                    # In this case a contour does not represent a hole
                    contour = list((point[0][0], point[0][1]) for point in contour)

                    # Split contour into subcontours that do not have self
                    # intersections.
                    subcontours = get_subcontours(contour)

                    for subcontour in subcontours:
                        if compute_scores:
                            # compute probability of the shape
                            mask = np.zeros(model_output.shape, dtype=np.uint8)
                            cv2.drawContours(
                                mask,
                                np.asarray([[[x, y]] for x, y in subcontour]),
                                contourIdx=-1,
                                color=1,
                                thickness=-1,
                            )
                            probability = cv2.mean(
                                current_label_soft_prediction, mask
                            )[0]

                        # convert the list of points to a closed polygon
                        points = [
                            Point(x=x / width, y=y / height) for x, y in subcontour
                        ]
                        polygon = Polygon(points=points)

                        if polygon.to_roi().width > 0 and polygon.to_roi().height > 0:
                            # Contour is a closed polygon with area > 0
                            score = probability if compute_scores else 0
                            annotations.append(
                                Annotation(
                                    shape=polygon,
                                    labels=[ScoredLabel.from_label(label, score)],
                                )
                            )
                        else:
                            # Contour is a closed polygon with area == 0
                            warnings.warn(
                                "The geometry of the segmentation map you are "
                                "converting is not fully supported. Polygons with a "
                                "area of zero will be removed.",
                                UserWarning,
                            )
                else:
                    # If contour hierarchy[3] != -1 then contour has a parent and
                    # therefore is a hole
                    # Do not allow holes in segmentation masks to be filled silently,
                    # but trigger warning instead
                    warnings.warn(
                        "The geometry of the segmentation map you are converting is "
                        "not fully supported. A hole was found and will be filled.",
                        UserWarning,
                    )

    return Prediction(annotations=annotations)
