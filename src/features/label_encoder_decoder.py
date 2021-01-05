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

from itertools import product, starmap
from functools import partial
from typing import List, Tuple

import numpy as np

def get_claim_map(
    n:int,
    source_locations: np.ndarray,
    bhw: Tuple[int, int, int],
    model_src_vals: List[np.ndarray],
) -> np.ndarray:

    def get_n_closest_sources_encode(
        n:int,                                  # max number of sources to include
        source_locations:np.ndarray,            # locations of sources as an array of y,x idxs
        ij:Tuple[int,int],                      # the index to retrieve the sources for
    ) -> Tuple[List[int], Tuple[int, int]]:     # The n sources ordered by proximity idxs and the idx ij
        distances = np.linalg.norm(source_locations-np.array(ij), axis=1)
        closest_n_sources = np.argsort(distances)[:n]

        if len(closest_n_sources) < n:
            closest_n_sources = np.pad(
                closest_n_sources,
                (0, n-len(closest_n_sources)),
                mode="constant",
                constant_values = len(model_src_vals),
            )

        assert closest_n_sources.shape[0] == n

        return (closest_n_sources, ij)


    def get_src_flx(
        scarlet_data:List[np.ndarray],                  # SCARLET data
        src_idxs:List[int],                             # idx of source in the SCARLET output
        ij:Tuple[List[np.ndarray], Tuple[int, int]],    # idx in image space
    ) -> np.ndarray:                                    # the source value at i,j in each of the bands
        i, j = ij
        # each element in this list is an array of the flux
        # values that belong to each source
        # [n, b, 1, 1]
        src_flux_values = None
        try:
            src_flux_values = np.array([scarlet_data[src_idx][:, i, j] for src_idx in src_idxs if (src_idx != len(scarlet_data))])
        except:
            print(src_idxs)
            print(len(scarlet_data))
            raise ValueError("")

        # this should be [n, b]
        if src_flux_values.shape[0] < len(src_idxs):
            src_flux_values = np.pad(
                src_flux_values,
                (
                    (0, len(src_idxs)-src_flux_values.shape[0]),
                    (0, 0)
                ),
                mode="constant",
                constant_values=0,
            )

        assert src_flux_values.shape[0]==len(src_idxs), f"{src_flux_values.shape}, {src_idxs}"
        assert src_flux_values.shape[1]==scarlet_data[0].shape[0], f"{src_flux_values.shape}, {scarlet_data[0].shape}"

        return (src_flux_values, ij)


    def update_image(
        output_array:np.ndarray,        # [h, w, b, n]
        normed_flux_vals:np.ndarray,    # [n, b]
        ij:Tuple[int, int],             # pixel location
    ) -> None:
        i, j = ij
        output_array[i, j, ...] = normed_flux_vals.T[:]


    def normed_combined_flux(
        src_flux_values:np.ndarray, # [n, bands]
        ij:Tuple[int, int]
    ) -> Tuple[List[np.ndarray], Tuple[int, int]]:

        # restrict flux to positive values
        src_flux_cmb = np.clip(np.array(src_flux_values), a_min=0, a_max=None) # [n, b]
        flux_norm = src_flux_cmb.sum(axis=0) # [b,] total flux for each band

        normed = src_flux_cmb / flux_norm

        try:
            normed[np.isnan(normed)] = 1 / src_flux_cmb.shape[0]
        except:
            print(src_flux_values)
            print(src_flux_values.shape)
            print(src_flux_cmb)
            print(src_flux_cmb.shape)
            raise ValueError()


        return (normed, ij)


    out_shape = list(model_src_vals[0].shape[1:]) + [model_src_vals[0].shape[0], n]
    output_array = np.zeros(out_shape, dtype=np.float32)


    get_n_src_f = partial(get_n_closest_sources_encode, n, source_locations)
    get_src_flx_f = partial(get_src_flx, model_src_vals)
    update_output_f = partial(update_image, output_array)

    img_shape = model_src_vals[0].shape[1:]
    idxs = product(range(img_shape[0]), range(img_shape[1]))
    n_srcs_per_pixel = map(get_n_src_f, idxs)
    src_flx_per_pixel = starmap(get_src_flx_f, n_srcs_per_pixel)
    normed_src_flx_per_pixel = starmap(normed_combined_flux, src_flx_per_pixel)

    for _ in starmap(update_output_f, normed_src_flx_per_pixel):pass

    return output_array


def get_claim_vector_image_and_map(
    source_locations: np.ndarray,
    bkg: np.ndarray,
    bhw: Tuple[int, int, int],
    model_src_vals: List[np.ndarray],
):
    # Updates claim_vector_image and claim_map_image in place
    def single_pixel_vector(
        claim_vector_image: np.ndarray,
        claim_map_image: np.ndarray,
        centers: np.ndarray,
        bkg:np.ndarray,
        i: int,
        j: int,
        b: int,
    ) -> None:

        ijb_src_flux = np.array([m[b, i, j] for m in model_src_vals])
        ijb_src_flux_mask = ijb_src_flux > 0

        if bkg[i, j, 0] > 0.9 or ijb_src_flux_mask.sum()==0:
            idxs = list(product([-1, 0, 1], [-1, 0, 1]))
            idxs.remove((0, 0))

            claim_vector_image[i, j, b, ...] = np.array(idxs)
            claim_map_image[i, j, b, ...] = np.array([1/8 for _ in range(8)])
        else:
            connected_idxs = list(product([i, i - 1, i + 1], [j, j - 1, j + 1]))
            connected_idxs.remove((i, j))
            connected_array = np.array(connected_idxs)

            ijb_normed_src_flux = (ijb_src_flux * ijb_src_flux_mask) / (
                ijb_src_flux * ijb_src_flux_mask
            ).sum()

            def closest_center(centers: np.array, flux_mask: np.ndarray, idx: np.ndarray):
                dist = np.linalg.norm(centers - idx, axis=1)
                masked_dist = np.where(flux_mask, dist, np.inf)
                return centers[np.argmin(masked_dist)]

            closest_f = partial(closest_center, centers, ijb_src_flux_mask)
            closest_sources = np.array(list(map(closest_f, connected_array)))
            claim_vector = connected_array - closest_sources  # [8]

            claim_vector_image[i, j, b, ...] = claim_vector

            def convert_to_claim_map(
                centers: np.ndarray, normed_flux: np.ndarray, src: np.ndarray
            ):
                return (
                    (src == centers).all(axis=1).astype(np.float32) * ijb_normed_src_flux
                ).sum()

            convert_to_map_f = partial(convert_to_claim_map, centers, ijb_normed_src_flux)
            raw_claim_map = np.array(list(map(convert_to_map_f, closest_sources)))
            claim_map = raw_claim_map / raw_claim_map.sum()

            claim_map_image[i, j, b, ...] = claim_map

    n_bands, height, width = bhw
    claim_vector_image = np.zeros([height, width, n_bands, 8, 2], dtype=np.float32)
    claim_map_image = np.zeros([height, width, n_bands, 8], dtype=np.float)

    src_ys, src_xs = np.nonzero(source_locations)
    centers = np.array([src_ys, src_xs]).T  # [n, 2]

    single_pixel_f = partial(
        single_pixel_vector, claim_vector_image, claim_map_image, centers, bkg
    )

    idxs = product(range(height), range(width), range(n_bands))

    for _ in starmap(single_pixel_f, idxs):
        pass

    return claim_vector_image, claim_map_image


# use peak_local_max?, its much faster
def non_maximum_suppression(kernel_size: int, threshold: float, image: np.ndarray):

    image[image < threshold] = 0
    pad = (kernel_size - 1) // 2
    padded = np.pad(image, pad)
    output = np.zeros_like(image)

    idxs = product(
        range(padded.shape[0] - kernel_size), range(padded.shape[1] - kernel_size)
    )

    def update_max(y, x):
        output[y, x] = padded[y : y + kernel_size, x : x + kernel_size].max()

    for _ in starmap(update_max, idxs):
        pass

    output[image != output] = 0

    return output


def get_sources(
    flux_image: np.ndarray,  # [h, w, b]
    claim_vectors: np.ndarray,  # [h, w, b, 8, 2]
    claim_maps: np.ndarray,  # [h, w, b, 8]
    background_map: np.ndarray,  # [h, w]
    center_of_mass: np.ndarray,  # [h, w]
) -> np.ndarray:  # [n, h, w, b]

    src_locations = non_maximum_suppression(7, 0.1, center_of_mass)  # [h, w]
    src_centers = np.stack(np.nonzero(src_locations), axis=1)  # [n, 2]
    n_srcs = src_centers.shape[0]

    height, width, bands = flux_image.shape
    src_image = np.zeros([n_srcs, height, width, bands], dtype=np.float32)

    def distribute_source_flux(i, j, b):
        if background_map[i, j] > 0.9:
            return

        adj_idxs = list(product([i, i - 1, i + 1], [j, j - 1, j + 1]))
        adj_idxs.remove((i, j))
        adj_idx_array = np.array(adj_idxs)

        pixel_claim_vectors = (
            adj_idx_array - claim_vectors[i, j, b, ...].copy()
        )  # [8, 2]
        pixel_claim_map = claim_maps[i, j, b, :].copy()  # [8,]
        pixel_flux = flux_image[i, j, b]  # [0,]

        def closest_center(k):  # k of 8
            dist = np.linalg.norm(src_centers - pixel_claim_vectors[k, :], axis=1)
            closest_center = np.argmin(dist)
            return closest_center

        pixel_claim_src_idxs = np.array(list(map(closest_center, range(8))))

        def nth_flux(i):
            claim_mask = pixel_claim_src_idxs == i  # [8,]
            claim = (pixel_claim_map * claim_mask).sum()  # [0,]
            return claim * pixel_flux  # [0,]

        src_separation = np.array(list(map(nth_flux, range(n_srcs))))  # [n, ]

        src_image[:, i, j, b] = src_separation

    idxs = tqdm(
        product(range(height), range(width), range(bands)), total=height * width * bands
    )

    for _ in starmap(distribute_source_flux, idxs):
        pass

    return src_image  # [n, h, w, b]
