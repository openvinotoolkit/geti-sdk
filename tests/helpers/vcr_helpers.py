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
import glob
import logging
import os
import shutil
import time

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .constants import CASSETTE_EXTENSION, CASSETTE_PATH, DUMMY_HOST


def are_cassettes_available() -> bool:
    """
    Checks that the VCR cassettes required to run the tests offline are available

    :return: True if the cassettes are available in the proper path, False otherwise
    """
    if not os.path.isdir(CASSETTE_PATH):
        return False
    if len(os.listdir(CASSETTE_PATH)) > 0:
        return True
    return False


def replace_host_name_in_cassettes(server_address: str) -> None:
    """
    This function searches for the server_address in all cassette files and
    replaces all occurrences of that address by 'dummy_host'. The cassette files are
    updated in place

    :param server_address: Server address to search for and replace
    """
    host_name = server_address.replace("https://", "").strip("/")
    logging.info(f"Removing host '{host_name}' from test cassettes...")
    tqdm_prefix = "Scrubbing cassettes"

    t_start = time.time()
    cassette_paths = glob.glob(os.path.join(CASSETTE_PATH, f"*.{CASSETTE_EXTENSION}"))

    with logging_redirect_tqdm(tqdm_class=tqdm):
        for cassette_path in tqdm(cassette_paths, desc=tqdm_prefix):
            replace_host_name_in_cassette(
                host_name=host_name, path_to_cassette_file=cassette_path
            )
    logging.info(
        f"Hostname scrubbed from {len(cassette_paths)} cassette files in "
        f"{1000*(time.time()-t_start)} seconds"
    )


def replace_host_name_in_cassette(host_name: str, path_to_cassette_file: str) -> None:
    """
    This function searches for the host_name in a target cassette file and
    replaces all occurrences of that address by 'dummy_host'. The cassette file is
    updated in place

    :param host_name: Host name to search for and replace
    :param path_to_cassette_file: Path to the cassette file to search in
    """
    path_to_scrubbed_cassette_file = path_to_cassette_file + "_new"

    with open(path_to_cassette_file, "rt") as read_file, open(
        path_to_scrubbed_cassette_file, "wt"
    ) as write_file:
        for index, line in enumerate(read_file.readlines()):
            line = line.replace(host_name, DUMMY_HOST)
            write_file.write(line)

    os.remove(path_to_cassette_file)
    shutil.move(src=path_to_scrubbed_cassette_file, dst=path_to_cassette_file)
