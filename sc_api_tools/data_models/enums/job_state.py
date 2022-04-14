from enum import Enum
from typing import List


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
    INACTIVE = 'inactive'

    def __str__(self) -> str:
        """
        Returns the string representation of the JobState instance

        :return: string containing the job state
        """
        return self.value

    @classmethod
    def active_states(cls) -> List['JobState']:
        """
        Returns a list of JobState instance which represent jobs that are still active

        :return: List of JobState instances
        """
        return [cls.IDLE, cls.PAUSED, cls.RUNNING]

    @classmethod
    def inactive_states(cls) -> List['JobState']:
        """
        Returns a list of JobState instance which represent jobs that are inactive,
        i.e. cancelled, errored or finished succesfully

        :return: List of JobState instances
        """
        return [cls.FINISHED, cls.ERROR, cls.FAILED, cls.FINISHED, cls.INACTIVE]
