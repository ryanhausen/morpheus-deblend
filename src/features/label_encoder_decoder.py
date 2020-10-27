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
from numpy.core.defchararray import startswith
from numpy.core.fromnumeric import prod


def get_claim_vector_image_and_map(
    source_locations:np.ndarray,
    bhw:Tuple[int, int, int],
    model_src_vals:List[np.ndarray],
):
    # Updates claim_vector_image and claim_map_image in place
    def single_pixel_vector(
        claim_vector_image:np.ndarray,
        claim_map_image:np.ndarray,
        centers:np.ndarray,
        i:int,
        j:int,
        b: int
    ) -> None:
        connected_idxs = list(product([i, i-1, i+1], [j, j-1, j+1]))
        connected_idxs.remove((i, j))
        connected_array = np.array(connected_idxs)

        ijb_src_flux = np.array([m[b,i,j] for m in model_src_vals])
        ijb_src_flux_mask = ijb_src_flux > 0

        ijb_normed_src_flux = (
            (ijb_src_flux * ijb_src_flux_mask)
            / (ijb_src_flux * ijb_src_flux_mask).sum()
        )

        def closest_center(
            centers:np.array,
            flux_mask:np.ndarray,
            idx:np.ndarray
        ):
            dist = np.linalg.norm(centers-idx, axis=1)
            masked_dist = np.where(flux_mask, dist, np.inf)
            return centers[np.argmin(masked_dist)]

        closest_f = partial(closest_center, centers, ijb_src_flux_mask)
        closest_sources = np.array(list(map(closest_f, connected_array)))
        claim_vector = connected_array - closest_sources # [8]

        claim_vector_image[i, j, b, ...] = claim_vector

        def convert_to_claim_map(
            centers:np.ndarray,
            normed_flux:np.ndarray,
            src:np.ndarray
        ):
            return ((src==centers).all(axis=1).astype(np.float32) * ijb_normed_src_flux).sum()

        convert_to_map_f = partial(convert_to_claim_map, centers, ijb_normed_src_flux)
        raw_claim_map = np.array(list(map(convert_to_map_f, closest_sources)))
        claim_map = raw_claim_map / raw_claim_map.sum()

        claim_map_image[i, j, b, ...] = claim_map


    n_bands, height, width = bhw
    claim_vector_image = np.zeros([height, width, n_bands, 8, 2], dtype=np.float32)
    claim_map_image = np.zeros([height, width, n_bands, 8], dtype=np.float)

    src_ys, src_xs = np.nonzero(source_locations)
    centers = np.array([src_ys, src_xs]).T # [n, 2]


    single_pixel_f = partial(
        single_pixel_vector,
        claim_vector_image,
        claim_map_image,
        centers
    )

    idxs = product(range(height), range(width), range(n_bands))

    for _ in starmap(single_pixel_f, idxs): pass

    return claim_vector_image, claim_map_image

# use peak_local_max?, its much faster
def non_maximum_suppression(kernel_size:int, threshold:float, image:np.ndarray):

    image[image<threshold] = 0
    pad = (kernel_size - 1) // 2
    padded = np.pad(image, pad)
    output = np.zeros_like(image)


    idxs = product(
        range(padded.shape[0]-kernel_size),
        range(padded.shape[1]-kernel_size)
    )

    def update_max(y, x):
        output[y, x] = padded[y:y+kernel_size, x:x+kernel_size].max()

    for _ in starmap(update_max, idxs): pass

    output[image!=output] = 0

    return output



def get_sources(
    flux_image: np.ndarray,     # [h, w, b]
    claim_vectors: np.ndarray,  # [h, w, b, 8, 2]
    claim_maps: np.ndarray,     # [h, w, b, 8]
    background_map: np.ndarray, # [h, w]
    center_of_mass: np.ndarray  # [h, w]
) -> np.ndarray: #[n, h, w, b]

    src_locations = non_maximum_suppression(7, 0.1, center_of_mass) # [h, w]
    src_centers = np.stack(np.nonzero(src_locations), axis=1) # [n, 2]
    n_srcs = src_centers.shape[0]


    height, width, bands = flux_image.shape
    src_image = np.zeros([n_srcs, height, width, bands], dtype=np.float32)

    def distribute_source_flux(i, j, b):
        if background_map[i, j] > 0.9:
            return

        adj_idxs = list(product([i, i-1, i+1], [j, j-1, j+1]))
        adj_idxs.remove((i,j))
        adj_idx_array = np.array(adj_idxs)

        pixel_claim_vectors = adj_idx_array - claim_vectors[i, j, b, ...].copy() # [8, 2]
        pixel_claim_map = claim_maps[i, j, b, :].copy() # [8,]
        pixel_flux = flux_image[i, j, b] # [0,]

        def closest_center(k): # k of 8
            dist = np.linalg.norm(src_centers-pixel_claim_vectors[k,:], axis=1)
            closest_center = np.argmin(dist)
            return closest_center

        pixel_claim_src_idxs = np.array(list(map(closest_center, range(8))))

        def nth_flux(i):
            claim_mask = pixel_claim_src_idxs == i # [8,]
            claim = (pixel_claim_map * claim_mask).sum() #[0,]
            return claim * pixel_flux # [0,]

        src_separation = np.array(list(map(nth_flux, range(n_srcs)))) #[n, ]

        src_image[:, i, j, b] = src_separation

    idxs = tqdm(product(range(height), range(width), range(bands)), total=height*width*bands)

    for _ in starmap(distribute_source_flux, idxs): pass

    return src_image #[n, h, w, b]

