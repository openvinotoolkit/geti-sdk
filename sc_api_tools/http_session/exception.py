from typing import Optional, Union, BinaryIO, Dict


class SCRequestException(Exception):
    def __init__(
            self,
            method: str,
            url: str,
            status_code: int,
            request_data: Dict[str, Union[dict, str, list, BinaryIO]],
            response_data: Optional[Union[dict, str, list]] = None
    ):
        """
        SCRequestException is raised upon unsuccessful requests to the SC cluster

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

        if response_data:
            self.response_message = response_data.get("message", None)
            self.response_error_code = response_data.get("error_code", None)

    def __str__(self) -> str:
        """
        String representation of the unsuccessful http request to the SC cluster
        """
        error_str = f"{self.method} request to '{self.url}' failed with status code " \
                    f"{self.status_code}."
        if self.response_error_code and self.response_message:
            error_str += f" Server returned error code '{self.response_error_code}' " \
                         f"with message '{self.response_message}'"
        return error_str
