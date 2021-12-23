from typing import Optional

import requests
import shutil
import zipfile
import os

from tqdm import tqdm


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
    print(f"Downloading {filename}...")
    with requests.get(url, stream=True) as r:
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


if __name__ == '__main__':
    # ------------------------------------
    # Configure target folder for download
    # ------------------------------------
    TARGET_DIR = 'data'

    ensure_directory_exists(TARGET_DIR)

    image_url = 'http://images.cocodataset.org/zips/val2017.zip'
    annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    # Download the zip files
    image_zip = download_file(image_url, TARGET_DIR)
    annotations_zip = download_file(annotations_url, TARGET_DIR)

    # Create directories for images and annotations
    image_dir = os.path.join(TARGET_DIR, 'images')
    annotations_dir = os.path.join(TARGET_DIR, 'annotations')
    ensure_directory_exists(image_dir)
    ensure_directory_exists(annotations_dir)
    zip_to_extraction_mapping = {image_zip: image_dir, annotations_zip: TARGET_DIR}

    # Extract images and annotations
    for zipfile_path, target_dir in zip_to_extraction_mapping.items():
        print(f'Extracting {zipfile_path}...')
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

    image_subset_names = os.listdir(image_dir)
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
