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
from geti_sdk.http_session import GetiRequestException


class TestGetiRequestException:
    def test_exception_string_representation(self):
        # Arrange
        response_message = "dummy_message"
        response_error_code = "dummy_error_code"
        url = "dummy_url"
        method = "POST"
        status_code = 400
        request_data = {"dummy": "request"}

        # Act
        exception = GetiRequestException(
            url=url,
            method=method,
            status_code=status_code,
            request_data=request_data,
            response_data={
                "message": response_message,
                "error_code": response_error_code,
            },
        )
        exception_string = f"{exception}"

        # Assert
        assert url in exception_string
        assert method in exception_string
        assert f"{status_code}" in exception_string
        assert response_error_code in exception_string
        assert response_message in exception_string
