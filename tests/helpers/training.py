import time
from typing import Optional, Union

from sc_api_tools.data_models import Job, Task
from sc_api_tools.http_session import SCRequestException
from sc_api_tools.rest_clients import TrainingClient

from .enums import SdkTestMode


def attempt_to_train_task(
    training_client: TrainingClient,
    task: Union[int, Task],
    test_mode: SdkTestMode = SdkTestMode.OFFLINE,
) -> Job:
    """
    Attempts to train the `task` (either a task or the index of the task in the list
    of trainable tasks for the project), by issuing a 'train_task' command from the
    training_client.

    If the initial train request fails with a 'project_not_train_ready' error code,
    this method will attempt to issue the command again (max 5 attempts) after
    a short sleep period.

    :param training_client: TrainingClient to use to train the task
    :param task: Task or index of task to be trained
    :param test_mode: SdkTestMode in which the training is attempted. If set to
        SdkTestMode.OFFLINE, no sleep period is used between attempts
    :return: Training job that was created by the training request
    """
    job: Optional[Job] = None
    not_ready_response = "project_not_train_ready"
    n_attempts = 5

    for i in range(n_attempts):
        try:
            job = training_client.train_task(task)
            break
        except SCRequestException as error:
            if error.response_error_code != not_ready_response:
                raise error
        if test_mode != SdkTestMode.OFFLINE:
            time.sleep(2)

    if job is not None:
        return job
    else:
        raise ValueError(
            f"Unable to start training on server, received '{not_ready_response}' on "
            f"each request for {n_attempts} calls in a row. Please make sure that the "
            f"project is ready to start training the task."
        )
