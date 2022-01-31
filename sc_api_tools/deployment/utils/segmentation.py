import numpy as np
import cv2


def mask_from_soft_prediction(
    soft_prediction: np.ndarray, threshold: float, blur_strength: int = 5
) -> np.ndarray:
    """
    Creates a hard prediction containing the final label index per pixel
    :param soft_prediction: Output from segmentation network. Assumes floating point
                            values, between 0.0 and 1.0. Can be a 2d-array of shape
                            (height, width) or per-class segmentation logits of shape
                            (height, width, num_classes)
    :param threshold: minimum class confidence for each pixel.
                            The higher the value, the more strict the segmentation is
                            (usually set to 0.5)
    :param blur_strength: The higher the value, the smoother the segmentation output
                            will be, but less accurate
    :return: Numpy array of the hard prediction
    """
    soft_prediction_blurred = cv2.blur(soft_prediction, (blur_strength, blur_strength))
    if len(soft_prediction.shape) == 3:
        # Apply threshold to filter out `unconfident` predictions, then get max along
        # class dimension
        soft_prediction_blurred[soft_prediction_blurred < threshold] = 0
        mask = np.argmax(soft_prediction_blurred, axis=2)
    elif len(soft_prediction.shape) == 2:
        # In the binary case, simply apply threshold
        mask = soft_prediction_blurred > threshold
    else:
        raise ValueError(
            f"Invalid prediction input of shape {soft_prediction.shape}. "
            f"Expected either a 2D or 3D array."
        )
    return mask
