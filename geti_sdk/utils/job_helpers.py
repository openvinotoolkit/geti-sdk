# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
import logging
import time
from typing import List, Optional

from geti_sdk.data_models.enums.job_state import JobState
from geti_sdk.data_models.job import Job
from geti_sdk.http_session import GetiRequestException, GetiSession
from geti_sdk.rest_converters.job_rest_converter import JobRESTConverter


def get_job_by_id(
    job_id: str, session: GetiSession, workspace_id: str
) -> Optional[Job]:
    """
    Retrieve Job information from the Intel® Geti™ server.

    :param job_id: Unique ID of the job to retrieve
    :param session: GetiSession instance addressing the Intel® Geti™ platform
    :param workspace_id: ID of the workspace in which the job was created
    :return: Job instance holding the details of the job
    """
    try:
        response = session.get_rest_response(
            url=f"workspaces/{workspace_id}/jobs/{job_id}", method="GET"
        )
    except GetiRequestException as error:
        if error.status_code == 404:
            return None
        else:
            raise error
    return JobRESTConverter.from_dict(response)


def get_job_with_timeout(
    job_id: str,
    session: GetiSession,
    workspace_id: str,
    job_type: str = "training",
    timeout: int = 15,
) -> Job:
    """
    Retrieve a Job from the Intel® Geti™ server, by it's unique ID. If the job is not
    found within the specified `timeout`, a RuntimeError is raised.

    :param job_id: Unique ID of the job to retrieve
    :param session: GetiSession instance addressing the Intel® Geti™ platform
    :param workspace_id: ID of the workspace in which the job was created
    :param job_type: String representing the type of job, for instance "training" or
        "testing"
    :param timeout: Time (in seconds) after which the job retrieval will timeout
    :raises: RuntimeError if the job is not found within the specified timeout
    :return: Job instance holding the details of the job
    """
    job = get_job_by_id(job_id=job_id, session=session, workspace_id=workspace_id)
    if job is not None:
        logging.info(
            f"{job_type.capitalize()} job with ID {job_id} retrieved successfully."
        )
    else:
        t_start = time.time()
        while job is None and (time.time() - t_start < timeout):
            logging.info(
                f"{job_type.capitalize()} job status could not be retrieved from the "
                f"platform yet. Re-attempting to fetch job status. Looking for job "
                f"with ID {job_id}"
            )
            time.sleep(2)
            job = get_job_by_id(
                session=session, job_id=job_id, workspace_id=workspace_id
            )
        if job is None:
            raise RuntimeError(
                f"Unable to find the resulting {job_type} job on the Intel® Geti™ "
                f"server."
            )
    job.workspace_id = workspace_id
    return job


def monitor_jobs(
    session: GetiSession, jobs: List[Job], timeout: int = 10000, interval: int = 15
) -> List[Job]:
    """
    Monitor and print the progress of all jobs in the list `jobs`. Execution is
    halted until all jobs have either finished, failed or were cancelled.

    Progress will be reported in 15s intervals

    :param session: GetiSession instance addressing the Intel® Geti™ platform
    :param jobs: List of jobs to monitor
    :param timeout: Timeout (in seconds) after which to stop the monitoring
    :param interval: Time interval (in seconds) at which the TrainingClient polls
        the server to update the status of the jobs. Defaults to 15 seconds
    :return: List of finished (or failed) jobs with their status updated
    """
    monitoring = True
    completed_states = [
        JobState.FINISHED,
        JobState.CANCELLED,
        JobState.FAILED,
        JobState.ERROR,
    ]
    logging.info("---------------- Monitoring progress -------------------")
    jobs_to_monitor = [job for job in jobs if job.status.state not in completed_states]
    try:
        t_start = time.time()
        t_elapsed = 0
        while monitoring and t_elapsed < timeout:
            msg = ""
            complete_count = 0
            for job in jobs_to_monitor:
                job.update(session)
                msg += (
                    f"{job.name}  -- "
                    f"  Phase: {job.status.user_friendly_message} "
                    f"  State: {job.status.state} "
                    f"  Progress: {job.status.progress:.1f}%"
                )
                if job.status.state in completed_states:
                    complete_count += 1
            if complete_count == len(jobs_to_monitor):
                break
            logging.info(msg)
            time.sleep(interval)
            t_elapsed = time.time() - t_start
    except KeyboardInterrupt:
        logging.info("Job monitoring interrupted, stopping...")
        for job in jobs:
            job.update(session)
        return jobs
    if t_elapsed < timeout:
        logging.info("All jobs completed, monitoring stopped.")
    else:
        logging.info(
            f"Monitoring stopped after {t_elapsed:.1f} seconds due to timeout."
        )
    return jobs
