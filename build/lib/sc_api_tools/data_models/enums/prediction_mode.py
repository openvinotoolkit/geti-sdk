from enum import Enum


class PredictionMode(Enum):
    """
    This Enum represents the mode used to generate predictions in SC
    """
    LATEST = 'latest'
    AUTO = 'auto'
    ONLINE = 'online'

    def __str__(self):
        """
        Returns the string representation of the PredictionMode instance
        :return:
        """
        return self.value
