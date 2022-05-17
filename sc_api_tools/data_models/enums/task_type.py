from enum import Enum


class TaskType(Enum):
    """
    This enum represents the different task types in SC
    """
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    ANOMALY_CLASSIFICATION = "anomaly_classification"
    ANOMALY_DETECTION = "anomaly_detection"
    ANOMALY_SEGMENTATION = "anomaly_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    ROTATED_DETECTION = "rotated_detection"
    DATASET = "dataset"
    CROP = "crop"

    def __str__(self) -> str:
        """
        Returns the string representation of the TaskType instance

        :return: string containing the task type
        """
        return self.value

    @property
    def is_trainable(self) -> bool:
        """
        Returns True if a task of this TaskType is trainable, False otherwise

        :return:
        """
        return self not in NON_TRAINABLE_TASK_TYPES

    @property
    def is_global(self) -> bool:
        """
        Returns True if a task of this TaskType produces global labels, False otherwise

        :return:
        """
        return self in GLOBAL_TASK_TYPES

    @property
    def is_local(self) -> bool:
        """
        Returns True if a task of this TaskType produces local labels, False otherwise

        :return:
        """
        return self not in GLOBAL_TASK_TYPES and self not in NON_TRAINABLE_TASK_TYPES

    @property
    def is_anomaly(self) -> bool:
        """
        Returns True if a task of this TaskType is an anomaly task, False otherwise

        :return:
        """
        return self in ANOMALY_TASK_TYPES

    @property
    def is_segmentation(self) -> bool:
        """
        Returns True if a task of this TaskType is an anomaly task, False otherwise

        :return:
        """
        return self in SEGMENTATION_TASK_TYPES

    @classmethod
    def from_domain(cls, domain: 'Domain'):
        """
        Instantiates a :py:class:`~sc_api_tools.data_models.enums.task_type.TaskType`
        from a given :py:class:`~sc_api_tools.data_models.enums.domain.Domain`

        :param domain: domain to get the TaskType for
        :return: TaskType instance corresponding to the `domain`
        """
        return cls[domain.name]


NON_TRAINABLE_TASK_TYPES = [TaskType.DATASET, TaskType.CROP]

ANOMALY_TASK_TYPES = [
    TaskType.ANOMALY_CLASSIFICATION,
    TaskType.ANOMALY_DETECTION,
    TaskType.ANOMALY_SEGMENTATION
]

GLOBAL_TASK_TYPES = [TaskType.CLASSIFICATION, TaskType.ANOMALY_CLASSIFICATION]

SEGMENTATION_TASK_TYPES = [
    TaskType.SEGMENTATION,
    TaskType.ANOMALY_SEGMENTATION,
    TaskType.INSTANCE_SEGMENTATION
]