from enum import Enum


class JobState(Enum):
    """
    This Enum represents the state of a job on the SC cluster
    """
    IDLE = 'idle'
    RUNNING = 'running'
    PAUSED = 'paused'
    FINISHED = 'finished'
    ERROR = 'error'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

    def __str__(self) -> str:
        """
        Returns the string representation of the JobState instance

        :return: string containing the job state
        """
        return self.value
