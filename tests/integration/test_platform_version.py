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

from geti_sdk.platform_versions import GetiVersion


class TestGetiVersion:
    SC_MVP = "1.0.0-release-20220129184214"
    SC_11 = "1.1.0-release-20220624125113"
    GETI_10 = "1.0.0-release-20221005164936"

    def test_version_parsing_and_comparison(self):
        """
        Test parsing the version from a version string, for different release versions
        of the Intel Geti platform. Also test comparisons between versions
        """
        mvp_version = GetiVersion(TestGetiVersion.SC_MVP)
        sc11_version = GetiVersion(TestGetiVersion.SC_11)
        geti10_version = GetiVersion(TestGetiVersion.GETI_10)

        assert mvp_version.is_sc_mvp and not mvp_version.is_geti
        assert (
            sc11_version.is_sc_1_1
            and not sc11_version.is_geti
            and not sc11_version.is_sc_mvp
        )
        assert geti10_version.is_geti and geti10_version.is_geti_1_0

        assert geti10_version > sc11_version
        assert sc11_version > mvp_version
        assert not mvp_version > geti10_version
