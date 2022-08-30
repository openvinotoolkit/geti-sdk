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

import os
import shutil
import zipfile
from enum import Enum
from typing import Dict, Optional

import requests
from tqdm import tqdm

DEFAULT_COCO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
)


class COCOSubset(Enum):
    """
    Enum representing a certain subset of the MS COCO dataset.
    """

    TRAIN2014 = "train2014"
    VAL2014 = "val2014"
    TEST2014 = "test2014"
    TEST2015 = "test2015"
    TRAIN2017 = "train2017"
    VAL2017 = "val2017"
    TEST2017 = "test2017"
    UNLABELED2017 = "unlabeled2017"

    def __str__(self):
        """
        Return string representation of the COCOSubset instance.
        """
        return self.value

    def get_annotations(self) -> Optional[str]:
        """
        Return the name of the annotation folder for the subset, if any.

        :return: String containing the name of the annotation folder for the subset.
            If subset does not have annotations, returns None.
        """
        if self == COCOSubset.VAL2017 or self == COCOSubset.TRAIN2017:
            return "trainval2017"
        elif self == COCOSubset.VAL2014 or self == COCOSubset.TRAIN2014:
            return "trainval2014"
        elif (
            self == COCOSubset.TEST2017
            or self == COCOSubset.TEST2015
            or self == COCOSubset.TEST2014
            or self == COCOSubset.UNLABELED2017
        ):
            return None


def get_proxies(url: str = "") -> Dict[str, str]:
    """
    Determine whether or not to use proxies to attempt to reach a certain url.

    :param url: URL that should be resolved
    :return:
    """
    print(f"Connecting to url {url}...")
    proxies: Dict[str, str] = {}
    try:
        requests.head(url=url, proxies=proxies, timeout=10)
        return proxies
    except requests.exceptions.ConnectionError:
        print(
            "Unable to reach URL for COCO dataset download, attempting to connect "
            "via proxy"
        )
    proxies = {
        "http": "http://proxy-mu.intel.com:911",
        "https": "http://proxy-mu.intel.com:912",
    }
    try:
        requests.head(url=url, proxies=proxies)
        print("Connection succeeded.")
    except requests.exceptions.ConnectionError as error:
        raise ValueError(
            "Unable to resolve URL with any proxy settings, please try to turn off "
            "your VPN connection before attempting again."
        ) from error
    return proxies


def download_file(
    url: str, target_folder: Optional[str], check_valid_archive: bool = False
) -> str:
    """
    Download a file from `url` to a folder on local disk `target_folder`.

    NOTE: If a file with the same name as the file to be downloaded already exists in
        `target_folder`, this function will not download anything. If
        `check_valid_archive` is True, this function not only checks if the target
        file exists but also if it is a valid .zip archive.

    :param url:
    :param target_folder:
    :return: path to the downloaded file
    """
    filename = url.split("/")[-1]
    if target_folder is None:
        target_folder = "data"
    path_to_file = os.path.join(target_folder, filename)
    if os.path.exists(path_to_file) and os.path.isfile(path_to_file):
        if check_valid_archive:
            if not zipfile.is_zipfile(path_to_file):
                print(
                    f"File {filename} exists at {path_to_file}, but is is not a valid "
                    f"archive. Overwriting the existing file."
                )
                shutil.rmtree(path_to_file)
        print(f"File {filename} exists at {path_to_file}. No new data was downloaded.")
        return path_to_file

    proxies = get_proxies(url)
    print(f"Downloading {filename}...")
    with requests.get(url, stream=True, proxies=proxies) as r:
        if r.status_code != 200:
            r.raise_for_status()
            raise RuntimeError(
                f"Request to {url} failed, returned status code {r.status_code}"
            )
        file_size = int(r.headers.get("Content-Length", 0))
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc="") as r_raw:
            with open(path_to_file, "wb") as f:
                shutil.copyfileobj(r_raw, f)
    print("Download complete.")
    return path_to_file


def ensure_directory_exists(directory_path: str):
    """
    Check that a directory exists at `directory_path`, and if not create it.

    :param directory_path:
    :return:
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def directory_has_coco_subset(target_folder: str, coco_subset: COCOSubset) -> bool:
    """
    Check if a certain directory contains a particular subset of the coco dataset.

    :param target_folder: Directory to check
    :param coco_subset: Subset to look for
    :return: True if the directory contains the subset, False otherwise
    """
    if not os.path.exists(target_folder):
        return False
    required_dirs = ["images", "annotations"]
    for directory in required_dirs:
        if directory not in os.listdir(target_folder):
            return False
    image_dir = os.path.join(target_folder, "images")
    subset_image_dir = os.path.join(image_dir, str(coco_subset))
    if not os.path.exists(subset_image_dir):
        return False
    if len(os.listdir(subset_image_dir)) == 0:
        return False
    annotations = coco_subset.get_annotations()
    if annotations is not None:
        annotations_dir_content = os.listdir(os.path.join(target_folder, "annotations"))
        if len(annotations_dir_content) == 0:
            return False
        for filename in annotations_dir_content:
            if f"instances_{str(coco_subset)}" in filename:
                return True
        return False
    return True


def get_coco_dataset_from_path(
    target_folder: str = "data",
    coco_subset: Optional[COCOSubset] = None,
    verbose: bool = False,
) -> str:
    """
    Check if a coco dataset exists in the directory at `target_folder`.
    If no such directory exists, this method will attempt to download the COCO
    dataset to the designated folder.

    :param target_folder: Folder to get the COCO dataset from, or to download the
        dataset into
    :param coco_subset: Optional Subset to download. If passed as None, the script
        will check if any of the coco subsets exists in the folder, and will not
        download any data if at least one subset is found.
    :param verbose: True to print verbose output, False for silent mode
    :return: Path to the COCO dataset
    """
    ensure_directory_exists(target_folder)
    found_subset = None

    if coco_subset is None:
        for subset in COCOSubset:
            if subset.get_annotations() is not None and directory_has_coco_subset(
                target_folder, subset
            ):
                found_subset = subset
                break
        if found_subset is None:
            found_subset = COCOSubset.VAL2017
    else:
        found_subset = coco_subset

    if directory_has_coco_subset(target_folder=target_folder, coco_subset=found_subset):
        if verbose:
            print(
                f"COCO dataset (subset: {str(found_subset)}) found at path "
                f"{target_folder}"
            )
        return target_folder
    else:
        if verbose:
            print(
                f"COCO dataset was not found at path {target_folder}, making an "
                f"attempt to download the data."
            )

    image_url = f"http://images.cocodataset.org/zips/{str(found_subset)}.zip"
    annotations_name = found_subset.get_annotations()
    if annotations_name is not None:
        annotations_url = (
            f"http://images.cocodataset.org/annotations/"
            f"annotations_{annotations_name}.zip"
        )
    else:
        if verbose:
            print(
                f"Unable to download annotations for COCO subset {found_subset}. "
                f"Downloading images only"
            )
        annotations_url = None

    # Download the zip files
    image_zip = download_file(image_url, target_folder, check_valid_archive=True)
    if annotations_url is not None:
        annotations_zip = download_file(
            annotations_url, target_folder, check_valid_archive=True
        )
    else:
        annotations_zip = None

    # Create directories for images and annotations
    image_dir = os.path.join(target_folder, "images")
    ensure_directory_exists(image_dir)
    zip_to_extraction_mapping = {image_zip: image_dir}

    if annotations_zip is not None:
        annotations_dir = os.path.join(target_folder, "annotations")
        ensure_directory_exists(annotations_dir)
        zip_to_extraction_mapping.update({annotations_zip: target_folder})
    else:
        annotations_dir = None

    # Extract images and annotations
    for zipfile_path, target_dir in zip_to_extraction_mapping.items():
        if verbose:
            print(f"Extracting {zipfile_path}...")
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

    image_subset_names = os.listdir(image_dir)
    if annotations_dir is not None:
        # Remove unused annotations
        for filename in os.listdir(annotations_dir):
            filepath = os.path.join(annotations_dir, filename)
            if "instances" not in filename:
                os.remove(filepath)
                continue
            annotation_subset = os.path.splitext(filename.split("_")[1])[0]
            if annotation_subset not in image_subset_names:
                os.remove(filepath)

    if verbose:
        print("COCO dataset downloaded and extracted successfully.")
    return target_folder


def get_coco_dataset(dataset_path: Optional[str] = None, verbose: bool = False) -> str:
    """
    Check if the COCO dataset is present at the specified path. If not,
    this method will attempt to download the dataset to the path specified.

    If no path is passed, this method will check or create the default path: the
    folder 'data' in the top level of the sc-api-tools package.

    :param dataset_path: Path to check against.
    :param verbose: True to print detailed output, False to check silently
    :return: Path to the COCO dataset
    """
    if dataset_path is None:
        dataset_path = DEFAULT_COCO_PATH
    return get_coco_dataset_from_path(dataset_path, verbose=verbose)
