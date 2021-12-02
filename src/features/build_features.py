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
"""Functions that transform the raw data into trainable data."""

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import product, starmap, takewhile
from multiprocessing import Pool
from typing import Callable, List, Tuple

import numpy as np
import scarlet
import scarlet.psf as psf
import sharedmem
from astropy.io import fits
from astropy.wcs import WCS
from scipy import signal
from skimage.transform import rescale, resize
from tqdm import tqdm

import src.features.scarlet_helper as scarlet_heplper
import src.features.label_encoder_decoder as label_encoder_decoder

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")
DATA_PATH_PROCESSED = os.path.join(DATA_PATH, "processed")
DATA_PATH_PROCESSED_SCARLET = os.path.join(DATA_PATH_PROCESSED, "scarlet")
DATA_PATH_PROCESSED_TRAIN = os.path.join(DATA_PATH_PROCESSED, "train")
DATA_PATH_PROCESSED_TEST = os.path.join(DATA_PATH_PROCESSED, "test")
DATA_PATH_RAW = os.path.join(DATA_PATH, "raw")

NUM_TRAIN_EXAMPLES = 2000
NUM_TEST_EXAMPLES = 500


def make_dirs():
    dirs = [
        DATA_PATH,
        DATA_PATH_PROCESSED,
        DATA_PATH_PROCESSED_TRAIN,
        DATA_PATH_PROCESSED_TEST,
        DATA_PATH_RAW,
    ]

    mk = lambda s: os.makedirs(s) if not os.path.exists(s) else None
    for _ in map(mk, dirs):  # apply func to each element
        pass


def transform_catalog(catalog_fname: str, mask_fname: str):
    header = fits.getheader(mask_fname)
    wcs = WCS(header)
    catalog_image = np.zeros([header["NAXIS2"], header["NAXIS1"], 5], dtype=np.float32)

    with open(catalog_fname, "r") as f:
        lines = f.readlines()
        cols = lines[32].strip().split(",")
        rows = [l.strip().split(",") for l in lines[33:]]

    def row_convert_f(cols: List[str], row: List[str]):

        _id = float(row[cols.index("3dhst_id")])

        ra = float(row[cols.index("ra")])
        dec = float(row[cols.index("dec")])
        [[x, y]] = wcs.all_world2pix([[ra, dec]], 0)

        morphology_cols = list(
            map(
                cols.index,
                ["spheroid", "disk", "irregular", "ps_compact", "background"],
            )
        )

        morphology = np.array([float(row[i]) for i in morphology_cols])

        return [_id, y, x, morphology]

    def plot_src(arr: np.ndarray, row: List[float]):
        _id, y, x, morphology = row
        arr[int(y), int(x), :] = morphology

    convert_f = partial(row_convert_f, cols)
    draw_f = partial(plot_src, catalog_image)

    for _ in map(draw_f, map(convert_f, rows)):
        pass

    out_path = os.path.join(DATA_PATH_PROCESSED, "catalog_data.fits")
    fits.PrimaryHDU(header=header, data=catalog_image).writeto(out_path, overwrite=True)

    return catalog_image


def validate_idx(
    mask: np.ndarray,
    validate_array: np.ndarray,
    upper_y: int,
    upper_x: int,
    img_size: int,
    yx: Tuple[int, int],
) -> bool:
    y, x = yx

    if y + img_size > upper_y or x + img_size > upper_x:
        return False

    slice_y = slice(y, y + img_size)
    slice_x = slice(x, x + img_size)

    # if validate_array[slice_y, slice_x].copy().sum() < 36:
    #     return False

    return mask[slice_y, slice_x].copy().all()


def idx_generator(start: int, end: int) -> int:
    while True:
        yield random.randint(start, end)


def make_idx_collection(
    mask: np.ndarray,
    validation_arr: np.ndarray,
    img_size: int,
    collection_size: int,
    start_y: int,
    end_y: int,
    start_x: int,
    end_x: int,
    train: bool,
) -> List[int]:

    # # FOR CATALOG LOCATIONS ====================================================
    # with open(os.path.join(DATA_PATH_PROCESSED, "interesting_sources"), "r") as f:
    #     lines = [l.replace("(", "").replace(")", "").replace("'", "").strip().split(",") for l in f.readlines()[1:]]

    # col_id, col_ra, col_dec, col_area, col_morph = 0, 1, 2, 3, 4

    # spheroids = list(filter(lambda r: r[col_morph].strip()=="spheroid", lines))
    # disks = list(filter(lambda r: r[col_morph].strip()=="disk", lines))
    # irregulars = list(filter(lambda r: r[col_morph].strip()=="irregular", lines))
    # ps_compacts = list(filter(lambda r: r[col_morph].strip()=="ps_compact", lines))

    # train_ratio = 0.8

    # def get_split(vals):
    #     idx = int(len(vals) * train_ratio)
    #     if train:
    #         return vals[:idx]
    #     else:
    #         return vals[idx:]

    # # 'mask' is a header now, but used to be array change back in calling
    # # function if using random idxs
    # wcs = WCS(mask)
    # def convert_f(vals):
    #     _id, ra, dec = vals[col_id].strip(), vals[col_ra].strip(), vals[col_dec].strip()
    #     [[_x, _y]] = wcs.all_world2pix([[float(ra), float(dec)]], 0)

    #     y = int(_y) - img_size//2
    #     x = int(_x) - img_size//2

    #     return (int(_id), (y, x))

    # all_srcs = (
    #     get_split(spheroids)
    #     + get_split(disks)
    #     + get_split(irregulars)
    #     + get_split(ps_compacts)
    # )

    # vals = list(map(convert_f, all_srcs))

    # return vals
    # # FOR CATALOG LOCATIONS ====================================================

    # FOR RANDOM SLICES ========================================================
    y_gen = idx_generator(start_y, end_y)
    x_gen = idx_generator(start_x, end_x)

    valid_count = lambda iyx: iyx[0] < collection_size
    valid_idx_f = partial(validate_idx, mask, validation_arr, end_y, end_x, img_size)

    # We want a collection of valid (y, x) coordinates that we can use to
    # crop from the larger image. Explanation starting from the right:
    #
    # zip(y_gen, x_gen) -- is an inifite generator that produces random integer
    #                      tuples of (y, x) within the start and end ranges
    #
    # filter(valid_idx_f, ...) -- filters tuple pairs from zip(y_gen, x_gen)
    #                             that can't be used to generate a valid sample
    #                             based on the conditions in valid_idx_f function
    #
    # enumerate(...) -- returns a tuple (i, (y, x)) where i is an integer that
    #                   counts the values as they are generated. In our case,
    #                   its a running count of the valid tuples generated
    #
    # takewhile(valid_count, ...) -- takewhile returns the values in the
    #                                collection while valid_count returns True.
    #                                In our case, valid_count returns true while
    #                                the i returned by enumerate(...) is less
    #                                than collection_size
    #
    # [iyx for iyx ...] -- iyx is a valid tuple (i, (y, x)) from takewhile(...)
    return [
        iyx
        for iyx in takewhile(
            valid_count, enumerate(filter(valid_idx_f, zip(y_gen, x_gen)))
        )
    ]
    # FOR RANDOM SLICES ========================================================


# Center of Mass Label Functions ===============================================
# https://stackoverflow.com/a/46892763/2691018
def gaussian_kernel_2d(kernlen, std=8):
    """Returns a 2D Gaussian kernel array."""

    if kernlen % 2 == 0:
        raise ValueError("Only odd kernel lengths are supported")

    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


# UPDATES 'image' in place
def insert_gaussian(image, g_kern, y, x) -> None:
    height, width = image.shape
    half_kernel_len = int(g_kern.shape[0] / 2)

    def image_slice_f(yx, bound):
        return slice(max(yx - half_kernel_len, 0), min(yx + half_kernel_len, bound),)

    def kernel_slice_f(yx, bound):
        begin = half_kernel_len - min(
            half_kernel_len, half_kernel_len - (half_kernel_len - yx)
        )

        end = half_kernel_len + min(half_kernel_len, bound - yx,)

        return slice(begin, end)

    image_ys = image_slice_f(y, height)
    image_xs = image_slice_f(x, width)

    kernel_ys = kernel_slice_f(y, height)
    kernel_xs = kernel_slice_f(x, width)

    tmp_image = image[image_ys, image_xs].copy()
    tmp_kernel = g_kern[kernel_ys, kernel_xs].copy()

    image[image_ys, image_xs] = np.maximum(tmp_image, tmp_kernel)


def build_center_mass_image(
    source_locations: np.ndarray, gaussian_kernel_len: int, gaussian_kernel_std: int,
) -> np.ndarray:
    center_of_mass = np.zeros_like(source_locations, dtype=np.float32)
    src_ys, src_xs = np.nonzero(source_locations)

    gaussian_kernel = gaussian_kernel_2d(gaussian_kernel_len, std=gaussian_kernel_std)

    insert_gaussian_f = partial(insert_gaussian, center_of_mass, gaussian_kernel)

    for _ in starmap(insert_gaussian_f, zip(src_ys, src_xs)):
        pass

    return center_of_mass[:, :, np.newaxis]


# ==============================================================================
# ==============================================================================

global_data = None


def crop_convert_and_save(  #                   0:4    4:8       8       9:
    data: np.ndarray,  # [height, width, 12] = flux + weights + bkg +  morph(source pixels)
    psfs: List[np.ndarray],
    wcs: WCS,
    img_size: int,
    save_dir: str,
    scarlet_dir: str,
    iyx: Tuple[int, Tuple[int, int]],
) -> None:
    i, (y, x) = iyx
    ys, xs = slice(y, y + img_size), slice(x, x + img_size)

    bands = ["h", "j", "v", "z"]
    if global_data is not None:
        flux = np.transpose(global_data[ys, xs, :4].copy(), axes=(2, 0, 1))  # [b, h, w]
        weights = np.transpose(
            global_data[ys, xs, 4:8].copy(), axes=(2, 0, 1)
        )  # [b, h, w]
        background = global_data[ys, xs, 8:9].copy()  # [h, w, 1]
        catalog_data = np.transpose(
            global_data[ys, xs, 9:].copy(), axes=(2, 0, 1)
        )  # [h, w, 5]

        source_locations = (catalog_data[0, :, :].copy() > 0).astype(np.int)
    else:
        flux = np.transpose(data[ys, xs, :4].copy(), axes=(2, 0, 1))  # [b, h, w]
        weights = np.transpose(data[ys, xs, 4:8].copy(), axes=(2, 0, 1))  # [b, h, w]
        background = data[ys, xs, 8:9].copy()  # [h, w, 1]
        catalog_data = np.transpose(
            data[ys, xs, 9:].copy(), axes=(2, 0, 1)
        )  # [h, w, 5]

        source_locations = (catalog_data[0, :, :].copy() > 0).astype(np.int)

    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * 4)

    scarlet_file_path = os.path.join(scarlet_dir, f"{i}.fits")
    if os.path.exists(scarlet_file_path):
        scarlet_src_vals = [arr for arr in fits.getdata(scarlet_file_path)]
    else:
        scarlet_src_vals = scarlet_heplper.get_scarlet_fit(
            bands, psfs, model_psf, flux, weights, catalog_data,
        )

        fits.PrimaryHDU(data=np.array(scarlet_src_vals)).writeto(scarlet_file_path)

    center_of_mass = build_center_mass_image(source_locations, 51, 8)

    ys, xs = np.nonzero(source_locations)
    source_idxs = np.array(list(zip(ys, xs)))

    # CLAIM MAP ONLY ===========================================================
    # claim_map_image = label_encoder_decoder.get_claim_map(
    #     5,
    #     source_idxs,
    #     flux.shape,
    #     scarlet_src_vals
    # ) # [h, w, b, n]
    # CLAIM MAP ONLY ===========================================================

    # CLAIM VECTOR V1 ==========================================================
    # (
    #     claim_vector_image,
    #     claim_map_image,
    # ) = label_encoder_decoder.get_claim_vector_image_and_map(
    #     source_locations, background, flux.shape, scarlet_src_vals
    # )
    # CLAIM VECTOR V1 ==========================================================

    # CLAIM VECTOR MAGNITUDES instance v3=======================================
    # (
    #     claim_vector_image,
    #     claim_map_image,
    # ) = label_encoder_decoder.get_claim_vector_image_and_map_discrete_directions(
    #     source_locations,
    #     background,
    #     flux.shape,
    #     scarlet_src_vals,
    # )
    # CLAIM VECTOR MAGNITUDES instance v3=======================================

    # CLAIM VECTOR MAGNITUDES instance v5=======================================
    # n = 5
    # (
    #     claim_vector_image,
    #     claim_map_image,
    # ) = label_encoder_decoder.get_n_closest_claim_vectors(
    #     source_locations,
    #     background,
    #     flux.shape,
    #     scarlet_src_vals,
    #     n,
    # )
    # CLAIM VECTOR MAGNITUDES instance v5=======================================

    # FLUX WEIGHTED CLAIM VECTOR instance v6====================================
    # n = 3
    # (
    #     claim_vector_image,
    #     claim_map_image,
    # ) = label_encoder_decoder.get_n_closest_fw_claim_vectors_maps(
    #     source_locations,
    #     background,
    #     flux.shape,
    #     scarlet_src_vals,
    #     n,
    # )
    # FLUX WEIGHTED CLAIM VECTOR instance v6====================================

    # AVG N CLOSEST AVG CMAP instance v7========================================
    # n = 5
    # (
    #     claim_vector_image,
    #     claim_map_image,
    # ) = label_encoder_decoder.get_n_closest_avg_claim_vector(
    #     source_locations,
    #     background,
    #     flux.shape,
    #     scarlet_src_vals,
    #     n,
    # )
    # AVG N CLOSEST AVG CMAP instance v7========================================

    # Limit bands, n closest vectors v8=========================================
    bands = 1
    n = 3
    (
        claim_vector_image,
        claim_map_image,
    ) = label_encoder_decoder.get_n_closest_claim_vector_map_limit_bands(
        source_locations, background, flux.shape, scarlet_src_vals, n, bands,
    )
    # Limit bands, n closest vectors v8=========================================

    save_data = [
        np.transpose(flux, axes=(1, 2, 0)),
        background,
        center_of_mass,
        claim_vector_image,
        claim_map_image,
    ]

    fname_prefix = lambda s: f"{i}-{s}.fits"
    save_names = list(
        map(
            fname_prefix,
            ["flux", "background", "center_of_mass", "claim_vectors", "claim_maps"],
        )
    )

    def save_f(name, arr):
        fits.PrimaryHDU(data=arr).writeto(os.path.join(save_dir, name))

    for _ in starmap(save_f, zip(save_names, save_data)):
        pass


def get_full_name(fname_key: str) -> str:
    return next(filter(lambda f: fname_key in f, os.listdir(DATA_PATH_RAW)))


def main(img_size: int) -> None:
    make_dirs()

    if (
        len(os.listdir(DATA_PATH_PROCESSED_TRAIN)) > 0
        or len(os.listdir(DATA_PATH_PROCESSED_TEST)) > 0
    ):
        print(f"Files exists in {DATA_PATH_PROCESSED} skipping data extraction.")
    else:
        random.seed(12171988)

        # CATALOG ==============================================================
        mask_fname = get_full_name("mask")
        data_catalog = transform_catalog(
            os.path.join(DATA_PATH_RAW, get_full_name("catalog")),
            os.path.join(DATA_PATH_RAW, mask_fname),
        )
        # CATALOG ==============================================================

        # BUILD INDEXES ========================================================
        with fits.open(os.path.join(DATA_PATH_RAW, mask_fname)) as mask_hdul:
            mask = mask_hdul[0].data  # pylint: disable=no-member
            val_array = data_catalog  # pylint: disable=no-member

            train_ys, train_xs = (11500, 21400), (3800, 19400)
            test_ys, test_xs = (3200, 11500), (3000, 18000)

            train_idxs = make_idx_collection(
                mask,
                val_array,
                img_size,
                NUM_TRAIN_EXAMPLES,
                *train_ys,
                *train_xs,
                True,
            )
            test_idxs = make_idx_collection(
                mask, val_array, img_size, NUM_TEST_EXAMPLES, *test_ys, *test_xs, False
            )
            del mask
        # BUILD INDEXES ========================================================

        # PSFs =================================================================

        def rescale_psf(band):
            arr = rescale(
                fits.getdata(os.path.join(DATA_PATH_RAW, "tinytim", f"{band}.fits")),
                27 / 73,
            )
            arr /= arr.sum()

            fits.PrimaryHDU(data=arr).writeto(
                os.path.join(DATA_PATH_RAW, "tinytim", f"{band}_resized.fits"),
                overwrite=True,
            )

        #        for _ in map(rescale_psf, ["v", "z"]):
        #            pass

        #        fname = lambda b: f"{b}.fits" if b in "hj" else f"{b}_resized.fits"
        fname = lambda b: f"{b}.fits"
        psf_path = lambda b: os.path.join(DATA_PATH_RAW, "tinytim", fname(b))
        psfs = np.array([fits.getdata(psf_path(b)) for b in "hjvz"])

        # PSFs =================================================================

        # FLUX, WEIGHTS and MORPHOLOGIES =======================================
        file_keywords = [
            "f160w_v2.0_sci",
            "f125w_v2.0_sci",
            "f606w_v2.0_sci",
            "f850lp_v2.0_sci",
            "f160w_v2.0_wht",
            "f125w_v2.0_wht",
            "f606w_v2.0_wht",
            "f850lp_v2.0_wht",
            "background",
        ]

        data_fnames = [
            os.path.join(DATA_PATH_RAW, get_full_name(fname_key))
            for fname_key in file_keywords
        ]

        big_array_fname = os.path.join(DATA_PATH_PROCESSED, "combined_array.dat")
        if not os.path.exists(big_array_fname):
            data = np.memmap(
                big_array_fname, dtype=np.float32, mode="w+", shape=(25000, 25000, 14)
            )

            def update_arr(i):
                data[:, :, i] = arr[:, :]
                del arr

            for _ in map(update_arr, tqdm(range(len(file_keywords)))):
                pass

            data[:, :, 9:] = data_catalog[:, :, :]

            del data

        data = np.memmap(
            big_array_fname, dtype=np.float32, mode="r", shape=(25000, 25000, 14)
        )

        del data_catalog

        # data_flux = [fits.getdata(d, memmap=True) for d in tqdm(data_fnames[:4])]
        # data_wht = [fits.getdata(d, memmap=True) for d in tqdm(data_fnames[4:-1])]
        # data_lbls = [fits.getdata(data_fnames[-1], memmap=True)]

        print("Done opening")
        # FLUX and MORPHOLOGIES ================================================

        # ======================================================================
        # Data is [height, width,
        #             [ H_flx, J_flx, V_flx, Z_flx,
        #               H_wht, J_wht, V_wht, Z_wht,
        #               bkg,
        #               sph,dsk,irr,psc,bkg
        #             ]
        #         ]
        # ======================================================================
        # data = np.concatenate([np.dstack(data_flux + data_wht + data_lbls), data_catalog], axis=-1)
        # del data_catalog
        # del data_flux
        # del data_wht
        # del data_lbls

        print("Getting header")
        header = fits.getheader(data_fnames[0])
        wcs = WCS(header)

        if True:
            global global_data
            global_data = data
            # mp_array = sharedmem.empty_like(data)
            # mp_array[:] = data[:]

            train_crop_f = partial(
                crop_convert_and_save,
                None,
                psfs,
                wcs,
                img_size,
                DATA_PATH_PROCESSED_TRAIN,
                os.path.join(DATA_PATH_PROCESSED_SCARLET, "train"),
            )

            with Pool(30) as p:
                p.map(
                    train_crop_f,
                    tqdm(
                        train_idxs,
                        desc="Making training examples",
                        total=NUM_TRAIN_EXAMPLES,
                    ),
                )

            test_crop_f = partial(
                crop_convert_and_save,
                None,
                psfs,
                wcs,
                img_size,
                DATA_PATH_PROCESSED_TEST,
                os.path.join(DATA_PATH_PROCESSED_SCARLET, "test"),
            )

            with Pool(30) as p:
                p.map(
                    test_crop_f,
                    tqdm(
                        test_idxs,
                        desc="Making testing examples",
                        total=NUM_TRAIN_EXAMPLES,
                    ),
                )
        else:

            # TODO: change crop and save, to crop convert and save
            train_crop_f = partial(
                crop_convert_and_save,
                data,
                psfs,
                wcs,
                img_size,
                DATA_PATH_PROCESSED_TRAIN,
            )

            for _ in map(
                train_crop_f,
                tqdm(
                    train_idxs,
                    desc="Making training examples",
                    total=NUM_TRAIN_EXAMPLES,
                ),
            ):
                pass

            test_crop_f = partial(
                crop_convert_and_save,
                data,
                psfs,
                wcs,
                img_size,
                DATA_PATH_PROCESSED_TEST,
            )

            for _ in map(
                test_crop_f,
                tqdm(
                    test_idxs, desc="Making testing examples", total=NUM_TEST_EXAMPLES
                ),
            ):
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=256)

    main(parser.parse_args().input_size)
