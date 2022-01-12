from enum import Enum


class OptimizationType(Enum):
    MO = 'MO'
    POT = 'POT'
    NNCF = 'NNCF'

    def __str__(self) -> str:
        """
        Returns the string representation of the OptimizationType instance

        :return: string containing the OptimizationType
        """
        return self.value
