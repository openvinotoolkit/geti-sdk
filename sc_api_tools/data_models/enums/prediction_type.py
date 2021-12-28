from enum import Enum


class PredictionType(Enum):
    LATEST = 'latest'
    AUTO = 'auto'
    ONLINE = 'online'

    def __str__(self):
        """
        Returns the string representation of the PredictionType instance
        :return:
        """
        return self.value
