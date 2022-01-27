from enum import Enum


class OptimizationType(Enum):
    """
    This Enum represents the optimization type for an OptimizedModel in SC
    """
    NNCF = 'NNCF'
    POT = 'POT'
    MO = 'MO'


    def __str__(self) -> str:
        """
        Returns the string representation of the OptimizationType instance

        :return: string containing the OptimizationType
        """
        return self.value
