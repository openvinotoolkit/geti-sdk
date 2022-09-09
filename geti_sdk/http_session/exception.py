# Copyright (C) 2022 Intel Corporation
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

from typing import BinaryIO, Dict, Optional, Union


class GetiRequestException(Exception):
    """
    Exception representing an unsuccessful http request to the GETi cluster.
    """

    def __init__(
        self,
        method: str,
        url: str,
        status_code: int,
        request_data: Dict[str, Union[dict, str, list, BinaryIO]],
        response_data: Optional[Union[dict, str, list]] = None,
    ):
        """
        Raise this exception upon unsuccessful requests to the GETi cluster.

        :param method: Method that was used for the request, e.g. 'POST' or 'GET', etc.
        :param url: URL to which the request was made
        :param status_code: HTTP status code returned for the request
        :param request_data: Data that was included with the request.
        :param response_data: Optional data that was returned in response to the
            request, if any
        """
        self.method = method
        self.url = url
        self.status_code = status_code
        self.request_data = request_data

        self.response_message: Optional[str] = None
        self.response_error_code: Optional[str] = None

        if response_data:
            self.response_message = response_data.get("message", None)
            self.response_error_code = response_data.get("error_code", None)

    def __str__(self) -> str:
        """
        Return string representation of the unsuccessful http request to the GETi cluster
        """
        error_str = (
            f"{self.method} request to '{self.url}' failed with status code "
            f"{self.status_code}."
        )
        if self.response_error_code and self.response_message:
            error_str += (
                f" Server returned error code '{self.response_error_code}' "
                f"with message '{self.response_message}'"
            )
        return error_str
