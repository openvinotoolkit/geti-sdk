from enum import Enum


class OpenvinoModelName(Enum):
    """
    This Enum represents the names of the supported Openvino model architectures that
    can be used for direct inference
    """
    SSD = 'ssd'
    OTE_CLASSIFICATION = 'ote_classification'
    SEGMENTATION = 'segmentation'
    BLUR_SEGMENTATION = 'blur_segmentation'
    ANOMALY_CLASSIFICATION = 'anomaly_classification'

    def __str__(self) -> str:
        """
        Returns the string representation of the OptimizationType instance

        :return: string containing the OptimizationType
        """
        return self.value
