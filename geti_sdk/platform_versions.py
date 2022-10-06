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
from packaging.version import InvalidVersion, Version


class GetiVersion:
    """
    Version identifier of the Intel Geti platform
    """

    _FIRST_GETI_HASH = "20220129184214"

    def __init__(self, version_string: str) -> None:
        """
        Initialize the version

        :param version_string: String representation of the version, as returned by
            the Geti platform
        """
        version_parts = version_string.split("-")
        if len(version_parts) == 3:
            base_version = Version(version_parts[0])
            tag = version_parts[1]
            version_hash = version_parts[2]
        elif len(version_parts) == 4:
            base_version = Version(version_parts[0] + "-" + version_parts[1])
            tag = version_parts[2]
            version_hash = version_parts[3]
        else:
            raise InvalidVersion(
                f"Unable to parse full platform version. Received '{version_string}'"
            )
        self.version = base_version
        self.tag = tag
        self.hash = version_hash

    def __gt__(self, other):
        """
        Return True if this GetiVersion instance is a later version than `other`

        :param other: GetiVersion object to compare with
        :raises: TypeError if `other` is not a GetiVersion instance
        :return: True if this instance corresponds to a later version of the Intel
            Geti platform than `other`
        """
        if not isinstance(other, GetiVersion):
            raise TypeError(
                f"Unsupported comparison operation, {other} is not a GetiVersion."
            )
        return self.hash > other.hash

    def __eq__(self, other):
        """
        Return True if this GetiVersion instance is equal to `other`

        :param other: GetiVersion object to compare with
        :raises: TypeError if `other` is not a GetiVersion instance
        :return: True if this instance is equal to the GetiVersion passed in `other`
        """
        if not isinstance(other, GetiVersion):
            raise TypeError(
                f"Unsupported comparison operation, {other} is not a GetiVersion."
            )
        return (
            self.hash == other.hash
            and self.tag == other.tag
            and self.version == other.version
        )

    @property
    def is_sc_mvp(self) -> bool:
        """
        Return True if the version corresponds to a platform on the SC MVP version of
        the software.
        """
        return self.version == Version("1.0.0") and self.hash <= self._FIRST_GETI_HASH

    @property
    def is_sc_1_1(self) -> bool:
        """
        Return True if the version corresponds to a platform on the SC v1.1 version of
        the software.
        """
        return self.version == Version("1.1.0") and self.hash <= self._FIRST_GETI_HASH

    @property
    def is_geti_1_0(self) -> bool:
        """
        Return True if the version corresponds to a platform on the Geti version 1.0 of
        the software.
        """
        return (
            self.version.base_version == Version("1.0.0").base_version
            and self.hash >= self._FIRST_GETI_HASH
        )

    @property
    def is_geti(self) -> bool:
        """
        Return True if the version corresponds to any version of the Geti platform.
        Return False if it corresponds to any SC version.
        """
        return self.version > Version("1.0.0b0") and self.hash >= self._FIRST_GETI_HASH
