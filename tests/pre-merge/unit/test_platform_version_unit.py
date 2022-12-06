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

from geti_sdk.platform_versions import (
    GETI_10_VERSION,
    GETI_11_VERSION,
    SC_11_VERSION,
    SC_MVP_VERSION,
)


class TestGetiVersion:
    def test_version_parsing_and_comparison(self):
        """
        Test parsing the version from a version string, for different release versions
        of the Intel Geti platform. Also test comparisons between versions
        """

        assert SC_MVP_VERSION.is_sc_mvp and not SC_MVP_VERSION.is_geti
        assert (
            SC_11_VERSION.is_sc_1_1
            and not SC_11_VERSION.is_geti
            and not SC_11_VERSION.is_sc_mvp
        )
        assert GETI_10_VERSION.is_geti and GETI_10_VERSION.is_geti_1_0

        assert GETI_10_VERSION > SC_11_VERSION
        assert SC_11_VERSION > SC_MVP_VERSION
        assert not SC_MVP_VERSION > GETI_10_VERSION
        assert GETI_10_VERSION < GETI_11_VERSION
        assert GETI_11_VERSION >= GETI_10_VERSION
