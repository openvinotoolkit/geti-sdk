from enum import Enum


class TaskType(Enum):
    """
    This enum represents the different task types in SC
    """
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    ANOMALY_CLASSIFICATION = "anomaly_classification"
    DATASET = "dataset"
    CROP = "crop"

    def __str__(self) -> str:
        """
        Returns the string representation of the TaskType instance

        :return: string containing the task type
        """
        return self.value


NON_TRAINABLE_TASK_TYPES = [TaskType.DATASET, TaskType.CROP]

ANOMALY_TASK_TYPES = [TaskType.ANOMALY_CLASSIFICATION]

GLOBAL_TASK_TYPES = [TaskType.CLASSIFICATION, TaskType.ANOMALY_CLASSIFICATION]