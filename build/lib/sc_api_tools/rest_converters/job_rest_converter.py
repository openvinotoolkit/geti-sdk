from typing import Dict, Any

from sc_api_tools.data_models import Job
from sc_api_tools.utils import deserialize_dictionary


class JobRESTConverter:
    """
    Class that handles conversion of SC REST output for jobs to objects, and vice-versa
    """

    @staticmethod
    def from_dict(job_dict: Dict[str, Any]) -> Job:
        """
        Creates a Job instance from the input dictionary passed in `job_dict`

        :param job_dict: Dictionary representing a job on the SC cluster, as returned
            by the /jobs endpoints
        :return: Job instance, holding the job data contained in job_dict
        """
        return deserialize_dictionary(job_dict, output_type=Job)
