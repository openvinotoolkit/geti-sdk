import os
import shutil
import zipfile
from enum import Enum
from typing import Optional, Dict

import requests
from tqdm import tqdm


class COCOSubset(Enum):
    """
    This Enum represents a certain subset of the MS COCO dataset
    """
    TRAIN2014 = 'train2014'
    VAL2014 = 'val2014'
    TEST2014 = 'test2014'
    TEST2015 = 'test2015'
    TRAIN2017 = 'train2017'
    VAL2017 = 'val2017'
    TEST2017 = 'test2017'
    UNLABELED2017 = 'unlabeled2017'

    def __str__(self):
        return self.value

    def get_annotations(self) -> Optional[str]:
        if self == COCOSubset.VAL2017 or self == COCOSubset.TRAIN2017:
            return 'trainval2017'
        elif self == COCOSubset.VAL2014 or self == COCOSubset.TRAIN2014:
            return 'trainval2014'
        elif (
                self == COCOSubset.TEST2017 or
                self == COCOSubset.TEST2015 or
                self == COCOSubset.TEST2014 or
                self == COCOSubset.UNLABELED2017
        ):
            return None


def get_proxies(url: str = "") -> Dict[str, str]:
    """
    Determines whether or not to use proxies to attempt to reach a certain url

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
        "https": "http://proxy-mu.intel.com:912"
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


def download_file(url: str, target_folder: Optional[str]) -> str:
    """
    Downloads a file from `url` to a folder on local disk `target_folder`.

    NOTE: If a file with the same name as the file to be downloaded already exists in
        `target_folder`, this function will not download anything.

    :param url:
    :param target_folder:
    :return: path to the downloaded file
    """
    filename = url.split('/')[-1]
    if target_folder is None:
        target_folder = 'data'
    path_to_file = os.path.join(target_folder, filename)
    if os.path.exists(path_to_file) and os.path.isfile(path_to_file):
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
        file_size = int(r.headers.get('Content-Length', 0))
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc="") as r_raw:
            with open(path_to_file, 'wb') as f:
                shutil.copyfileobj(r_raw, f)
    print("Download complete.")
    return path_to_file


def ensure_directory_exists(directory_path: str):
    """
    Checks that a directory exists at `directory_path`, and if not will create it

    :param directory_path:
    :return:
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def directory_has_coco_subset(target_folder: str, coco_subset: COCOSubset) -> bool:
    """
    Checks if a certain directory contains a particular subset of the coco dataset

    :param target_folder: Directory to check
    :param coco_subset: Subset to look for
    :return: True if the directory contains the subset, False otherwise
    """
    if not os.path.exists(target_folder):
        return False
    required_dirs = ['images', 'annotations']
    for directory in required_dirs:
        if directory not in os.listdir(target_folder):
            return False
    image_dir = os.path.join(target_folder, 'images')
    subset_image_dir = os.path.join(image_dir, str(coco_subset))
    if not os.path.exists(subset_image_dir):
        return False
    if len(os.listdir(subset_image_dir)) == 0:
        return False
    annotations = coco_subset.get_annotations()
    if annotations is not None:
        annotations_dir_content = os.listdir(os.path.join(target_folder, 'annotations'))
        if len(annotations_dir_content) == 0:
            return False
        for filename in annotations_dir_content:
            if f'instances_{str(coco_subset)}' in filename:
                return True
        return False
    return True


def get_coco_dataset(
        target_folder: str = 'data',
        coco_subset: COCOSubset = COCOSubset.VAL2017
) -> str:
    """
    This method checks if a coco dataset exists in the directory at `target_folder`.
    If no such directory exists, this method will attempt to download the COCO
    dataset to the designated folder.

    :param target_folder: Folder to get the COCO dataset from, or to download the
        dataset into
    :param coco_subset: Subset to download
    :return: Path to the COCO dataset
    """
    ensure_directory_exists(target_folder)
    if directory_has_coco_subset(target_folder=target_folder, coco_subset=coco_subset):
        print(
            f"COCO dataset (subset: {str(coco_subset)}) found at path {target_folder}"
        )
        return target_folder
    else:
        print(
            f"COCO dataset was not found at path {target_folder}, making an attempt to "
            f"download the data."
        )

    image_url = f'http://images.cocodataset.org/zips/{str(coco_subset)}.zip'
    annotations_name = coco_subset.get_annotations()
    if annotations_name is not None:
        annotations_url = f'http://images.cocodataset.org/annotations/annotations_{annotations_name}.zip'
    else:
        print(
            f"Unable to download annotations for COCO subset {coco_subset}. "
            f"Downloading images only"
        )
        annotations_url = None

    # Download the zip files
    image_zip = download_file(image_url, target_folder)
    if annotations_url is not None:
        annotations_zip = download_file(annotations_url, target_folder)
    else:
        annotations_zip = None

    # Create directories for images and annotations
    image_dir = os.path.join(target_folder, 'images')
    ensure_directory_exists(image_dir)
    zip_to_extraction_mapping = {image_zip: image_dir}

    if annotations_zip is not None:
        annotations_dir = os.path.join(target_folder, 'annotations')
        ensure_directory_exists(annotations_dir)
        zip_to_extraction_mapping.update({annotations_zip: target_folder})
    else:
        annotations_dir = None

    # Extract images and annotations
    for zipfile_path, target_dir in zip_to_extraction_mapping.items():
        print(f'Extracting {zipfile_path}...')
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

    image_subset_names = os.listdir(image_dir)
    if annotations_dir is not None:
        # Remove unused annotations
        for filename in os.listdir(annotations_dir):
            filepath = os.path.join(annotations_dir, filename)
            if 'instances' not in filename:
                os.remove(filepath)
                continue
            annotation_subset = os.path.splitext(filename.split('_')[1])[0]
            if annotation_subset not in image_subset_names:
                os.remove(filepath)

    print("COCO dataset downloaded and extracted successfully.")
    return target_folder
