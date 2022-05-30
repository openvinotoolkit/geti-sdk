from enum import Enum


class JobType(Enum):
    """
    This Enum represents the type of a job on the SC cluster
    """
    UNDEFINED = 'undefined'
    TRAIN = 'train'
    INFERENCE = 'inference'
    RECONSTRUCT_VIDEO = 'reconstruct_video'
    EVALUATE = 'evaluate'
    OPTIMIZATION = 'optimization'

    def __str__(self) -> str:
        """
        Returns the string representation of the JobType instance

        :return: string containing the job type
        """
        return self.value
