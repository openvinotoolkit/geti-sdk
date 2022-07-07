from typing import Union

import cv2
import numpy as np
from PIL import Image as PILImage
from IPython.display import display

from sc_api_tools.data_models import Image


def simulate_low_light_image(
    image: Union[np.ndarray, Image], reduction_factor: float = 0.5
) -> np.ndarray:
    """
    Simulate a reduced intensity and exposure time for an input image.
    It does so by reducing the image brightness and adding simulated shot noise to the
    image.

    :param image: Original image
    :param reduction_factor: Brightness reduction factor. Should be in the
        interval [0, 1]
    :return: Copy of the image, simulated for low lighting conditions (lower
        intensity and exposure time).
    """
    if isinstance(image, np.ndarray):
        new_image = image.copy()
    elif isinstance(image, Image):
        new_image = image.numpy.copy()
    else:
        raise TypeError(f"Unsupported image type '{type(image)}'")

    # Reduce brightness
    new_image = cv2.convertScaleAbs(new_image, alpha=reduction_factor, beta=0)

    # Add some shot noise to simulate reduced exposure time
    PEAK = 255*reduction_factor
    new_image_with_noise = np.clip(np.random.poisson(new_image / 255.0 * PEAK) / PEAK * 255, 0, 255).astype('uint8')
    return new_image_with_noise


def display_image_in_notebook(image: Union[np.ndarray, Image], bgr: bool = True):
    """
    Display an image inline in a Juypter notebook.

    :param image: Image to display
    :param bgr: True if the image has its channels in BGR order, False if it
        is in RGB order. Defaults to True
    """
    if isinstance(image, np.ndarray):
        new_image = image.copy()
    elif isinstance(image, Image):
        new_image = image.numpy.copy()
    else:
        raise TypeError(f"Unsupported image type '{type(image)}'")

    if bgr:
        result = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    else:
        result = new_image
    img = PILImage.fromarray(result)
    display(img)
