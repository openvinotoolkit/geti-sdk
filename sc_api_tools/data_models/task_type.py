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

    @property
    def is_trainable(self):
        """
        Returns True if this instance of TaskType represents a trainable task

        :return: True if the task type is trainable, False otherwise
        """
        return self not in NON_TRAINABLE_TASK_TYPES

    def __str__(self) -> str:
        """
        Returns the string representation of the TaskType instance

        :return: string containing the task type
        """
        return self.value


NON_TRAINABLE_TASK_TYPES = [TaskType.DATASET, TaskType.CROP]

ANOMALY_TASK_TYPES = [TaskType.ANOMALY_CLASSIFICATION]
