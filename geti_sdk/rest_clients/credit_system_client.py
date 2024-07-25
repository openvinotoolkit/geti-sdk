# Copyright (C) 2024 Intel Corporation
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

from typing import List, Optional, Union

from geti_sdk.data_models import Subscription
from geti_sdk.data_models.job import Job, JobCost
from geti_sdk.http_session import GetiSession
from geti_sdk.utils.job_helpers import get_job_with_timeout
from geti_sdk.utils.serialization_helpers import deserialize_dictionary
from geti_sdk.utils.workspace_helpers import get_default_workspace_id


class CreditSystemClient:
    """
    Class to work with credits in Intel Geti.
    """

    def __init__(self, session: GetiSession):
        self.session = session
        self.workspace_id = get_default_workspace_id(self.session)
        # Make sure the Intel Geti Platform supports Credit System.
        self._is_supported = self.is_supported()

    def is_supported(self) -> bool:
        """
        Check if the Intel Geti Platform supports Credit system.
        """
        r = self.session.get_rest_response(
            url=self.session.base_url + "balance",
            method="GET",
        )
        if isinstance(r, dict):
            # If the Platform responds with the information about the available subscriptions,
            # then it supports Credit System. Session will decode the message into a `dict`
            return True
        else:
            # In case the server returns an empty 200 message, it means the Credit System is not supported.
            # The session would return a Response object.
            return False

    def get_balance(self) -> Optional[int]:
        """
        Get the current credit balance in the workspace.
        """
        response = self.session.get_rest_response(
            url=self.session.base_url + "balance",
            method="GET",
        )
        return response.get("available", None)

    def get_job_cost(self, job: Union[Job, str]) -> Optional[JobCost]:
        """
        Get the cost of a job.
        """
        if isinstance(job, Job):
            job_id = job.id
        else:
            job_id = job
        job = get_job_with_timeout(job_id, self.session, self.workspace_id)
        return job.cost

    def get_subscription(self) -> Optional[List[Subscription]]:
        """
        Get the subscription details for the workspace.
        """
        response = self.session.get_rest_response(
            url=self.session.base_url + "subscriptions",
            method="GET",
        )
        return [
            deserialize_dictionary(sub, Subscription)
            for sub in response.get("subscriptions", [])
        ]
