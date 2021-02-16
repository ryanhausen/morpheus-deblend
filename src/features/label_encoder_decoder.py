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

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

from tqdm import tqdm

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


# ==============================================================================
# Discretize claim vector directions
# ==============================================================================
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


# ENCODER vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def get_claim_vector_magnitudes_single_pixel(
    neighborhood_vectors: np.ndarray,
    claim_vector_magnitude: np.ndarray,
    claim_map: np.ndarray,
    model_vals: List[np.ndarray],
    src_centers: np.ndarray,
    y: int,
    x: int,
    b: int
) -> None:
    relative_vectors = src_centers - np.array([y, x])
    src_fluxes = np.array([max(model_vals[i][b, y, x], 0) for i in range(len(model_vals))])
    max_flux = src_fluxes.max()
    normed_flux = src_fluxes / max_flux if max_flux > 0 else src_fluxes

    flx_sum = src_fluxes.sum()
    uniform_dist = np.ones_like(src_fluxes) / src_fluxes.shape[0]
    normed_sum_to_one = src_fluxes / src_fluxes.sum() if flx_sum > 0 else uniform_dist

    cosine_measure = cosine_similarity(neighborhood_vectors, relative_vectors)

    euclidean_distance = euclidean_distances(neighborhood_vectors, relative_vectors)
    euclidean_norm = np.maximum(euclidean_distance.max(axis=1, keepdims=True), 1e-5)
    normed_euclidean_distance = euclidean_distance / euclidean_norm

    metric = cosine_measure * (1 - normed_euclidean_distance) * (normed_flux[np.newaxis, :])

    closest_srcs = np.argmax(metric, axis=1)
    selected_srcs = relative_vectors[closest_srcs, :]

    _claim_magnitudes = (selected_srcs * neighborhood_vectors).sum(axis=1)


    idxs, counts = np.unique(closest_srcs, return_counts=True)
    coefs = np.reciprocal(counts.astype(np.float32))
    _claim_map = np.array(list(map(
        lambda i: coefs[idxs==i][0] * normed_sum_to_one[i],
        closest_srcs
    )))

    claim_vector_magnitude[y, x, b, :] = _claim_magnitudes
    claim_map[y, x, b, :] = _claim_map

def get_claim_vector_image_and_map_discrete_directions(
    source_locations: np.ndarray,
    bkg: np.ndarray,
    bhw: Tuple[int, int, int],
    model_src_vals: List[np.ndarray],
):
    b, h, w = bhw
    idxs = product(range(h), range(w), range(b))

    neighborhood_vectors = np.array(list(product([0, -1, 1], [0, -1, 1]))[1:], dtype=np.float32)
    neighborhood_vectors /= np.linalg.norm(neighborhood_vectors, axis=-1)[:, np.newaxis]

    claim_vector_magnitude = np.zeros([h, w, b, 8], dtype=np.float32)
    claim_map = np.zeros([h, w, b, 8], dtype=np.float32)

    src_ys, src_xs = np.nonzero(source_locations)
    src_centers = np.array([src_ys, src_xs]).T  # [n, 2]

    encode_f = partial(
        get_claim_vector_magnitudes_single_pixel,
        neighborhood_vectors,
        claim_vector_magnitude,
        claim_map,
        model_src_vals,
        src_centers
    )

    for _ in starmap(encode_f, idxs):
        pass

    return claim_vector_magnitude, claim_map
# ENCODER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# DECODER vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def decode_discrete_vectors_single_pixel(
    output:np.ndarray, # [n, h, w, b]
    neighborhood_vectors: np.ndarray, # [8, 2]
    flux:np.ndarray, # [h, w, b]
    claim_vector_magnitude:np.ndarray, # [h, w, b, 8]
    claim_map:np.ndarray, # [h, w, b, 8]
    src_centers:np.ndarray, # [n, 2]
    y:int,
    x:int,
    b:int
) -> None:
    pixel_flux = flux[y, x, b]
    pixel_magnitudes = claim_vector_magnitude[y, x, b, :].copy()
    pixel_claim_map = claim_map[y, x, b, :].copy()

    relative_vectors = neighborhood_vectors * pixel_magnitudes[:, np.newaxis]
    relative_centers = src_centers - np.array([y, x])

    distances = euclidean_distances(relative_vectors, relative_centers) # [n_neighborhood, n_centers]
    closest_src = np.argmin(distances, axis=1)

    distributed_flux = pixel_flux * pixel_claim_map

    def update_output(src_idx:int, flx:float):
        output[src_idx, y, x, b] += flx

    for _ in starmap(update_output, zip(closest_src, distributed_flux)):
        pass


def get_sources_discrete_directions(
    flux_image: np.ndarray,  # [h, w, b]
    claim_vector_magnitude: np.ndarray,  # [h, w, b, 8]
    claim_map: np.ndarray,  # [h, w, b, 8]
    background_map: np.ndarray,  # [h, w]
    center_of_mass: np.ndarray,  # [h, w]
    bkg_thresh_coef: float = 0.7,
) -> np.ndarray: # [n, h, w, b]
    y, x, b = flux_image.shape

    src_locations = non_maximum_suppression(7, 0.1, center_of_mass)  # [h, w]
    src_centers = np.stack(np.nonzero(src_locations), axis=1) + 0.5 # [n, 2]

    output = np.zeros([src_centers.shape[0], y, x, b], dtype=np.float32)

    neighborhood_vectors = np.array(list(product([0, -1, 1], [0, -1, 1]))[1:], dtype=np.float32)
    neighborhood_vectors /= np.linalg.norm(neighborhood_vectors, axis=-1)[:, np.newaxis]

    idxs = product(range(y), range(x), range(b))

    decode_f = partial(
        decode_discrete_vectors_single_pixel,
        output,
        neighborhood_vectors,
        flux_image,
        claim_vector_magnitude,
        claim_map,
        src_centers
    )

    for _ in starmap(decode_f, idxs):
        pass

    #filter out background pixels
    #bkg_filter = background_map[np.newaxis, :, :, np.newaxis] > bkg_thresh_coef
    #return output * bkg_filter
    return output
# DECODER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ==============================================================================
# Discretize claim vector directions
# ==============================================================================




# ==============================================================================
# Closest n-sources claim vector
# ==============================================================================
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

# ENCODER vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def get_n_closest_claim_vectors_single_pixel(
    claim_vectors: np.ndarray,     # [h, w, b, n, 2]
    claim_map: np.ndarray,         # [h, w, b, n]
    model_vals: List[np.ndarray],
    src_centers: np.ndarray,       # [n, 2]
    n: int,
    y: int,
    x: int,
    b: int,
) -> None:
    relative_vectors = src_centers - np.array([y, x]) # [n_srcs, 2]
    relative_distances = np.linalg.norm(relative_vectors, axis=-1) # [n_srcs,]
    raw_closest_sources = np.argsort(relative_distances)[:n] # [n, ]

    num_pad = n - raw_closest_sources.shape[0]
    if num_pad > 0:
        n_closest_sources = np.pad(raw_closest_sources, (0, num_pad), mode="edge")
    else:
        n_closest_sources = raw_closest_sources

    selected_srcs = relative_vectors[n_closest_sources]


    src_fluxes = np.array([max(model_vals[i][b, y, x], 0) for i in raw_closest_sources])

    sum_flux = src_fluxes.sum()
    if sum_flux > 0:
        normed_flux = src_fluxes / sum_flux
    else:
        raw_n = raw_closest_sources.shape[0]
        normed_flux = np.ones([raw_n], dtype=np.float32) / raw_n

    idxs, counts = np.unique(n_closest_sources, return_counts=True)
    coefs = np.reciprocal(counts.astype(np.float32))

    claim = np.array(list(map(
        lambda i: coefs[idxs==i][0] * normed_flux[i==raw_closest_sources][0],
        n_closest_sources
    )))

    claim_vectors[y, x, b, ...] = selected_srcs
    claim_map[y, x, b, ...] = claim


def get_n_closest_claim_vectors(
    source_locations: np.ndarray,
    bkg: np.ndarray,
    bhw: Tuple[int, int, int],
    model_src_vals: List[np.ndarray],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]: # [h, w, b, n], [h, w, b, n, 2]

    b, y, x = bhw

    src_ys, src_xs = np.nonzero(source_locations)
    src_centers = np.array([src_ys, src_xs]).T  # [n, 2]

    idxs = product(range(y), range(x), range(b))

    claim_vector = np.zeros([y, x, b, n, 2], dtype=np.float32)
    claim_map = np.zeros([y, x, b, n], dtype=np.float32)

    encode_f = partial(
        get_n_closest_claim_vectors_single_pixel,
        claim_vector,
        claim_map,
        model_src_vals,
        src_centers,
        n,
    )

    for _ in starmap(encode_f, idxs):
        pass

    return claim_vector, claim_map

# ENCODER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# DECODER vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def decode_n_closest_sources_single_pixel(
    output:np.ndarray,
    flux:np.ndarray,
    claim_vector:np.ndarray,
    claim_map:np.ndarray,
    src_centers:np.ndarray,
    y:int,
    x:int,
    b:int,
) -> None:
    pixel_flux = flux[y, x, b]
    pixel_vectors = claim_vector[y, x, b, ...].copy() # [n, 2]
    pixel_claim_map = claim_map[y, x, b, ...].copy() # [n,]

    relative_centers = src_centers - np.array([y, x])

    distances = euclidean_distances(pixel_vectors, relative_centers) #[n, n_src_centers]
    closest_srcs = np.argmin(distances, axis=1)

    distributed_flux = pixel_flux * pixel_claim_map

    def update_output(src_idx:int, flx:float):
        output[src_idx, y, x, b] += flx

    for _ in starmap(update_output, zip(closest_srcs, distributed_flux)):
        pass


def decode_n_closest_sources(
    flux:np.ndarray,
    claim_vector:np.ndarray,
    claim_map:np.ndarray,
    src_centers:np.ndarray,
    background_map: np.ndarray,  # [h, w]
    center_of_mass: np.ndarray,  # [h, w]
    bkg_thresh_coef: float = 0.7,
) -> np.ndarray:

    src_locations = non_maximum_suppression(7, 0.1, center_of_mass)  # [h, w]
    src_centers = np.stack(np.nonzero(src_locations), axis=1) + 0.5 # [n, 2]

    y, x, b = flux.shape
    output = np.zeros([src_centers.shape[0], y, x, b], dtype=np.float32)

    idxs = product(range(y), range(x), range(b))

    decode_f = partial(
        decode_n_closest_sources_single_pixel,
        output,
        flux,
        claim_vector,
        claim_map,
        src_centers,
    )

    for _ in starmap(decode_f, tqdm(idxs, total=y*x*b)):
        pass

    return output

# DECODER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ==============================================================================
# Closest n-sources claim vector
# ==============================================================================

# ==============================================================================
# Closest flux-weighted n-sources claim vector
# ==============================================================================
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

# ENCODER vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def get_n_closest_fw_claim_vectors_single_pixel(
    claim_vectors: np.ndarray,      # [h, w, b, n, 2]
    claim_map: np.ndarray,          # [h, w, b, n]
    model_vals: List[np.ndarray],   # list(n)
    src_centers: np.ndarray,        # [n, 2]
    n: int,
    y: int,
    x: int,
    b: int,
) -> None:
    relative_vectors = src_centers - np.array([y, x]) # [n_srcs, 2]
    relative_distances = np.linalg.norm(relative_vectors, axis=-1) # [n_srcs,]
    normed_distances = relative_distances / relative_distances.max() # [n_srcs, ]

    src_fluxes = np.array([max(model_vals[i][b, y, x], 0) for i in range(len(model_vals))]) # [n_srcs, ]
    max_flux = src_fluxes.max()

    if max_flux <= 0:
        normed_flux = np.ones([src_fluxes.shape[0]]) / src_fluxes.shape[0] # [n_srcs, ]
        normed_sum_to_one = np.ones([src_fluxes.shape[0]]) / src_fluxes.shape[0] # [n_srcs, ]
    else:
        normed_flux = src_fluxes / src_fluxes.max() # [n_srcs, ]
        normed_sum_to_one = src_fluxes / src_fluxes.sum() # [n_srcs, ]

    metric = (1 - normed_distances) * normed_flux # [n_srcs, ]

    top_srcs = np.argsort(-metric)[:n] # [min(n, n_srcs), ]

    num_pad = n - top_srcs.shape[0]
    if num_pad > 0:
        n_closest_sources = np.pad(top_srcs, (0, num_pad), mode="edge") # [n, ]
    else:
        n_closest_sources = top_srcs # [n, ]


    selected_srcs = relative_vectors[n_closest_sources] # [n, 2]

    src_fluxes = np.array([max(model_vals[i][b, y, x], 0) for i in top_srcs]) # [min(n, n_srcs), ]

    sum_flux = src_fluxes.sum()
    if sum_flux > 0:
        normed_flux = src_fluxes / sum_flux # [min(n, n_srcs), ]
    else:
        normed_flux = np.ones([src_fluxes.shape[0]], dtype=np.float32) / n # [min(n, n_srcs), ]

    idxs, counts = np.unique(n_closest_sources, return_counts=True) # [min(n, n_srcs), ], [min(n, n_srcs), ]
    coefs = np.reciprocal(counts.astype(np.float32)) # [min(n, n_srcs), ]

    claim = np.array(list(map(
        lambda i: coefs[idxs==i][0] * normed_flux[i==top_srcs][0],
        n_closest_sources
    )))

    claim_vectors[y, x, b, ...] = selected_srcs
    claim_map[y, x, b, ...] = claim

def get_n_closest_fw_claim_vectors_maps(
    source_locations: np.ndarray,
    bkg: np.ndarray,
    bhw: Tuple[int, int, int],
    model_src_vals: List[np.ndarray],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]: # [h, w, b, n], [h, w, b, n, 2]
    b, y, x = bhw

    src_ys, src_xs = np.nonzero(source_locations)
    src_centers = np.array([src_ys, src_xs]).T  # [n, 2]

    idxs = product(range(y), range(x), range(b))

    claim_vector = np.zeros([y, x, b, n, 2], dtype=np.float32)
    claim_map = np.zeros([y, x, b, n], dtype=np.float32)

    encode_f = partial(
        get_n_closest_fw_claim_vectors_single_pixel,
        claim_vector,
        claim_map,
        model_src_vals,
        src_centers,
        n,
    )

    for _ in starmap(encode_f, idxs):
        pass

    return claim_vector, claim_map
# ENCODER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# DECODER vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def decode_n_closest_fw_sources_single_pixel(
    output:np.ndarray,
    flux:np.ndarray,
    claim_vector:np.ndarray,
    claim_map:np.ndarray,
    src_centers:np.ndarray,
    y:int,
    x:int,
    b:int,
) -> None:
    pixel_flux = flux[y, x, b]
    pixel_vectors = claim_vector[y, x, b, ...].copy() # [n, 2]
    pixel_claim_map = claim_map[y, x, b, ...].copy() # [n,]

    relative_centers = src_centers - np.array([y, x])

    distances = euclidean_distances(pixel_vectors, relative_centers) #[n, n_src_centers]
    closest_srcs = np.argmin(distances, axis=1)

    distributed_flux = pixel_flux * pixel_claim_map

    def update_output(src_idx:int, flx:float):
        output[src_idx, y, x, b] += flx

    for _ in starmap(update_output, zip(closest_srcs, distributed_flux)):
        pass

def decode_n_closest_fw_sources(
    flux:np.ndarray,
    claim_vector:np.ndarray,
    claim_map:np.ndarray,
    src_centers:np.ndarray
) -> np.ndarray:
    y, x, b = flux.shape
    output = np.zeros([src_centers.shape[0], y, x, b], dtype=np.float32)

    idxs = product(range(y), range(x), range(b))

    decode_f = partial(
        decode_n_closest_fw_sources_single_pixel,
        output,
        flux,
        claim_vector,
        claim_map,
        src_centers,
    )

    for _ in starmap(decode_f, tqdm(idxs, total=y*x*b)):
        pass

    return output
# DECODER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ==============================================================================
# Closest flux-weighted n-sources claim vector
# ==============================================================================





# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ==============================================================================
# Closest n-sources claim vector avg map
# ==============================================================================
# ENCODER vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def get_n_closest_avg_claim_vector_single_pixel(
    claim_vectors: np.ndarray,      # [h, w, n, 2]
    claim_map: np.ndarray,          # [h, w, n]
    model_vals: List[np.ndarray],   # list(n)
    src_centers: np.ndarray,        # [n, 2]
    n: int,
    y: int,
    x: int,
) -> None:
    relative_vectors = src_centers - np.array([y, x]) # [n_srcs, 2]
    relative_distances = np.linalg.norm(relative_vectors, axis=-1) # [n_srcs,]
    raw_closest_sources = np.argsort(relative_distances)[:n] # [n, ]

    num_pad = n - raw_closest_sources.shape[0]
    if num_pad > 0:
        n_closest_sources = raw_closest_sources
    else:
        n_closest_sources = np.pad(raw_closest_sources, (0, num_pad), mode="edge")

    selected_srcs = relative_vectors[n_closest_sources]

    claim_vectors[y, x, ...] = selected_srcs

    def get_normed_src_fluxes(band:int):
        src_fluxes = np.array([max(model_vals[i][band, y, x], 0) for i in raw_closest_sources])

        sum_flux = src_fluxes.sum()
        if sum_flux > 0:
            normed_flux = src_fluxes / sum_flux
        else:
            normed_flux = np.ones([n], dtype=np.float32) / n

        return normed_flux

    n_bands = model_vals[0].shape[0]
    avg_flux_contrib = np.array(list(map(get_normed_src_fluxes, range(n_bands)))).mean(axis=0)
    normed_avg_flux_contrib = avg_flux_contrib / avg_flux_contrib.sum()

    idxs, counts = np.unique(n_closest_sources, return_counts=True)
    coefs = np.reciprocal(counts.astype(np.float32))



    claim = np.array(list(map(
        lambda i: coefs[idxs==i][0] * normed_avg_flux_contrib[i==raw_closest_sources][0],
        n_closest_sources
    )))

    claim_map[y, x, ...] = claim

def get_n_closest_avg_claim_vector(
    source_locations: np.ndarray,
    bkg: np.ndarray,
    bhw: Tuple[int, int, int],
    model_src_vals: List[np.ndarray],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]: # [h, w, n], [h, w, n, 2]
    _, y, x = bhw

    src_ys, src_xs = np.nonzero(source_locations)
    src_centers = np.array([src_ys, src_xs]).T  # [n, 2]

    idxs = product(range(y), range(x))

    claim_vector = np.zeros([y, x, n, 2], dtype=np.float32)
    claim_map = np.zeros([y, x, n], dtype=np.float32)

    encode_f = partial(
        get_n_closest_avg_claim_vector_single_pixel,
        claim_vector,
        claim_map,
        model_src_vals,
        src_centers,
        n,
    )

    for _ in starmap(encode_f, idxs):
        pass

    claim_vector, claim_map
# ENCODER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# DECODER vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def decode_n_closest_avg_claim_vector_single_pixel(
    output:np.ndarray,
    flux:np.ndarray,
    claim_vector:np.ndarray,
    claim_map:np.ndarray,
    src_centers:np.ndarray,
    y:int,
    x:int,
    b:int,
) -> None:
    pixel_flux = flux[y, x, b]
    pixel_vectors = claim_vector[y, x, ...].copy() # [n, 2]
    pixel_claim_map = claim_map[y, x, ...].copy() # [n,]

    relative_centers = src_centers - np.array([y, x])

    distances = euclidean_distances(pixel_vectors, relative_centers) #[n, n_src_centers]
    closest_srcs = np.argmin(distances, axis=1)

    distributed_flux = pixel_flux * pixel_claim_map

    def update_output(src_idx:int, flx:float):
        output[src_idx, y, x, b] += flx

    for _ in starmap(update_output, zip(closest_srcs, distributed_flux)):
        pass


def decode_n_closest_fw_sources(
    flux:np.ndarray,
    claim_vector:np.ndarray,
    claim_map:np.ndarray,
    src_centers:np.ndarray
) -> np.ndarray:
    y, x, b = flux.shape
    output = np.zeros([src_centers.shape[0], y, x, b], dtype=np.float32)

    idxs = product(range(y), range(x), range(b))

    decode_f = partial(
        decode_n_closest_avg_claim_vector_single_pixel,
        output,
        flux,
        claim_vector,
        claim_map,
        src_centers,
    )

    for _ in starmap(decode_f, tqdm(idxs, total=y*x*b)):
        pass

    return output
# DECODER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ==============================================================================
# Closest n-sources claim vector avg map
# ==============================================================================




# ==============================================================================
# Closest n-sources claim vector limit bands
# ==============================================================================
# ENCODER vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def get_n_closest_claim_vector_map_limit_bands_single_pixel(
    claim_vectors: np.ndarray,     # [h, w, n, 2]
    claim_map: np.ndarray,         # [h, w, b, n]
    model_vals: List[np.ndarray],
    src_centers: np.ndarray,       # [n_srcs, 2]
    n: int,
    n_bands: int,
    y: int,
    x: int,
) -> None:
    # Claim vectors ============================================================
    relative_vectors = src_centers - np.array([y, x]) # [n_srcs, 2]
    relative_distances = np.linalg.norm(relative_vectors, axis=-1) # [n_srcs,]
    raw_closest_sources = np.argsort(relative_distances)[:n] # [min(n_srcs, n), ]

    num_pad = n - raw_closest_sources.shape[0]
    if num_pad > 0:
        n_closest_sources = np.pad(raw_closest_sources, (0, num_pad), mode="edge") # [n,]
    else:
        n_closest_sources = raw_closest_sources # [n,]

    selected_srcs = relative_vectors[n_closest_sources] # [n, 2]

    claim_vectors[y, x, ...] = selected_srcs
    # Claim vectors ============================================================

    # Claim maps ===============================================================
    raw_n = raw_closest_sources.shape[0]
    def get_normed_src_fluxes(band:int):
        src_fluxes = np.array([max(model_vals[i][band, y, x], 0) for i in raw_closest_sources])

        sum_flux = src_fluxes.sum()
        if sum_flux > 0:
            normed_flux = src_fluxes / sum_flux
        else:
            normed_flux = np.ones([raw_n], dtype=np.float32) / raw_n

        return normed_flux

    band_normed_flux = np.array(list(map(get_normed_src_fluxes, range(n_bands)))) # [n_bands, min(n_src, n)]

    if num_pad > 0:
        padded_band_normed_flux = np.pad(band_normed_flux, ((0, 0), (0, num_pad)), mode="edge")
    else:
        padded_band_normed_flux = band_normed_flux

    idxs, counts = np.unique(n_closest_sources, return_counts=True) # [min(n_srcs, n), ], [min(n_srcs, n), ]
    coefs = np.reciprocal(counts.astype(np.float32)) # [min(n_srcs, n), ]

    coef_map = np.array([coefs[idxs==i][0] for i in n_closest_sources])[np.newaxis, :] #[1, n]

    try:
        claim = padded_band_normed_flux * coef_map
    except:
        print("raw_closest_sources: ", raw_closest_sources.shape)
        print("selected_srcs_shape: ", selected_srcs.shape)
        print("band_normed_flux: ", band_normed_flux.shape)
        print("padded_band_normed_flux: ", padded_band_normed_flux.shape)
        print("coefs: ", coefs.shape)
        print("coef_map: ", coef_map.shape)
        raise ValueError("Things Broke! Oh Man!")


    claim_map[y, x, ...] = claim
    # Claim maps ===============================================================


def get_n_closest_claim_vector_map_limit_bands(
    source_locations: np.ndarray,
    bkg: np.ndarray,
    bhw: Tuple[int, int, int],
    model_src_vals: List[np.ndarray],
    n: int,
    n_bands:int,
) -> Tuple[np.ndarray, np.ndarray]: # [h, w, bands, n], [h, w, bands, n, 2]

    b, y, x = bhw

    src_ys, src_xs = np.nonzero(source_locations)
    src_centers = np.array([src_ys, src_xs]).T  # [n, 2]

    idxs = product(range(y), range(x))

    claim_vector = np.zeros([y, x, n, 2], dtype=np.float32)
    claim_map = np.zeros([y, x, n_bands, n], dtype=np.float32)

    encode_f = partial(
        get_n_closest_claim_vector_map_limit_bands_single_pixel,
        claim_vector,
        claim_map,
        model_src_vals,
        src_centers,
        n,
        n_bands,
    )

    for _ in starmap(encode_f, idxs):
        pass

    return claim_vector, claim_map
# ENCODER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# DECODER vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def decode_n_closest_claim_vector_map_limit_bands_single_pixel(
    output:np.ndarray,
    flux:np.ndarray,
    claim_vector:np.ndarray,
    claim_map:np.ndarray,
    src_centers:np.ndarray,
    y:int,
    x:int,
    b:int,
) -> None:
    pixel_flux = flux[y, x, b]
    pixel_vectors = claim_vector[y, x, b, ...].copy() # [n, 2]
    pixel_claim_map = claim_map[y, x, b, ...].copy() # [n,]

    relative_centers = src_centers - np.array([y, x])

    distances = euclidean_distances(pixel_vectors, relative_centers) #[n, n_src_centers]
    closest_srcs = np.argmin(distances, axis=1)

    distributed_flux = pixel_flux * pixel_claim_map

    def update_output(src_idx:int, flx:float):
        output[src_idx, y, x, b] += flx

    for _ in starmap(update_output, zip(closest_srcs, distributed_flux)):
        pass

def decode_n_closest_claim_vector_map_limit_bands(
    flux:np.ndarray,
    claim_vector:np.ndarray,
    claim_map:np.ndarray,
    src_centers:np.ndarray
) -> np.ndarray:
    y, x, b = flux.shape
    output = np.zeros([src_centers.shape[0], y, x, b], dtype=np.float32)

    idxs = product(range(y), range(x), range(b))

    decode_f = partial(
        decode_n_closest_fw_sources_single_pixel,
        output,
        flux,
        claim_vector,
        claim_map,
        src_centers,
    )

    for _ in starmap(decode_f, tqdm(idxs, total=y*x*b)):
        pass

    return output
# DECODER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ==============================================================================
# Closest n-sources claim vector limit bands
# ==============================================================================





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
