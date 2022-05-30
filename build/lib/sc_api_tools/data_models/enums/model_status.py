from enum import Enum


class ModelStatus(Enum):
    NOT_READY = 'NOT_READY'
    WEIGHTS_INITIALIZED = 'WEIGHTS_INITIALIZED'
    SUCCESS = 'SUCCESS'
    FAILED = 'FAILED'
    NOT_IMPROVED = 'NOT_IMPROVED'

    def __str__(self) -> str:
        """
        Returns the string representation of the ModelStatus instance

        :return: string containing the model status
        """
        return self.value
