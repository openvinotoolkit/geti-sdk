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
import pytest

from geti_sdk.http_session import GetiSession
from geti_sdk.http_session.geti_session import INITIAL_HEADERS


class TestGetiSession:
    def test_authenticate(self, fxt_geti_session: GetiSession):
        """
        Test that the authenticated GetiSession instance contains authentication cookies
        """
        fxt_geti_session.authenticate(verbose=False)

    def test_product_version(self, fxt_geti_session: GetiSession):
        """
        Test that the 'version' attribute of the session is assigned a valid product
        version
        """
        version_tests = [
            fxt_geti_session.version.is_sc_mvp,
            fxt_geti_session.version.is_sc_1_1,
            fxt_geti_session.version.is_geti_1_0,
            fxt_geti_session.version.is_geti_1_1,
        ]
        assert sum(version_tests) == 1

    @pytest.mark.vcr()
    def test_logout(self, fxt_geti_session: GetiSession):
        """
        Test that logging out of the platform works, and clears all cookies and headers
        """
        fxt_geti_session.logout()
        assert len(fxt_geti_session.cookies) == 0
        assert len(fxt_geti_session.headers) == len(INITIAL_HEADERS)
        for key, value in fxt_geti_session._cookies.items():
            assert value is None
