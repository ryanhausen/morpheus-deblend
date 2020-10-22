# MIT License
# Copyright 2020 Ryan Hausen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Functions that download and extract raw data."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from itertools import starmap
from tarfile import TarFile
from typing import Tuple

import requests
from tqdm import tqdm

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")
DATA_MANIFEST = os.path.join(DATA_PATH, "data_manifest.json")
DATA_PATH_RAW = os.path.join(DATA_PATH, "raw")


def save_response_content(response: requests.Response, destination: str, pbar_loc: int):
    CHUNK_SIZE = 32768

    pbar_iter = tqdm(
        response.iter_content(CHUNK_SIZE),
        position=pbar_loc,
        desc=f" Downloading {os.path.split(destination)[1]}",
    )

    def write_chunk(f_obj, chunk):
        f_obj.write(chunk)
        return f_obj

    f = open(destination, "wb")

    reduce(write_chunk, filter(None, pbar_iter), f)


# based on https://stackoverflow.com/a/39225272
def download_manifest_item(manifest_item_job: Tuple[int, Tuple[str, str]]) -> None:
    pbar_loc, (item, url) = manifest_item_job

    if os.path.exists(os.path.join(DATA_PATH_RAW, item)):
        print(os.path.join(DATA_PATH_RAW, item), " exists skipping.")
        return
    else:
        print("Downloading ", os.path.join(DATA_PATH_RAW, item))
        session = requests.Session()

        response = session.get(url, stream=True)

        is_token_cookie = lambda kv: kv[0].startswith("download_warning")
        get_cookie_value = lambda kv: kv[1]

        download_token = next(
            map(get_cookie_value, filter(is_token_cookie, response.cookies.items())),
            None,
        )

        if download_token:
            save_response_content(
                session.get(url, params={"confirm": download_token}),
                os.path.join(DATA_PATH_RAW, item),
                pbar_loc,
            )
        else:
            save_response_content(response, os.path.join(DATA_PATH_RAW, item), pbar_loc)


def extract_manifest_item(manifest_item_job: Tuple[str, str]) -> None:
    (item, _) = manifest_item_job

    tarball_fname = os.path.join(DATA_PATH_RAW, item)

    if os.path.exists(tarball_fname):
        with TarFile.open(tarball_fname, mode="r") as tar:
            member = tar.getmembers()[0]
            print(f"Extracting {member.name}. Size: {member.size/1e6}MB")
            tar.extract(member, DATA_PATH_RAW)

        print(f"Removing old archive {item}")
        os.remove(tarball_fname)
    else:
        print(f"")


def main():
    # get raw data items
    with open(DATA_MANIFEST, "r") as f:
        manifest = json.load(f).items()

    if not os.path.exists(DATA_PATH_RAW):
        os.makedirs(DATA_PATH_RAW)

    # Download data ============================================================
    print("Downloading raw data...")
    with ThreadPoolExecutor() as executor:
        for _ in executor.map(download_manifest_item, enumerate(manifest)):
            pass
    print("\n" * len(manifest))
    print("Done downloading data")
    # Download data ============================================================

    # Extract data =============================================================
    print("Extracting data... (may take a while)")
    with ThreadPoolExecutor() as executor:
        for _ in executor.map(extract_manifest_item, manifest):
            pass
    print("Done extracting data")
    # Extract data =============================================================


if __name__ == "__main__":
    main()
