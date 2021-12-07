from functools import partial
from itertools import product, starmap
from typing import Callable, Tuple

import numpy as np
from numpy.lib.utils import source
import skimage.feature as ski
from skimage.measure import label, regionprops
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

def encode(
    source_locations: np.ndarray,
    scarlet_images: np.ndarray,
    background: np.ndarray,
    bkg_threshold:float,
    n: int,
    display_progress: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes scarlet deblended sources into the PCR representation
    Args:
        source_locations (np.ndarray): Shape [N, 2]. The locations of the
                                       sources associated with the
                                       `scarlet_images`. IMPORTANT these
                                       need to be in the same order as the
                                       `scarlet_images`
        n (int): Number of nearest sources to encode
        display_progress (bool): if True display progress using tqdm
    Returns:
        A Tuple where the first element is the contribution_vectors and the
        second element is the contribution_maps
    """

    b, h, w = scarlet_images.shape[1:]

    contribution_vectors = np.zeros([h, w, n, 2], dtype=np.float32)
    contribution_maps = np.zeros([h, w, b, n], dtype=np.float32)

    encode_f = partial(
        _encode_single_pixel,
        contribution_vectors,
        contribution_maps,
        scarlet_images,
        source_locations,
    )

    indices = (
        tqdm(product(range(h), range(w)))
        if display_progress
        else product(range(h), range(w))
    )
    any(starmap(encode_f, indices))  # `any` will force iterating over the iterable

    # use background to filter claim vectors/claim maps
    bkg_mask = background>bkg_threshold
    src_mask = np.logical_not(bkg_mask)
    contribution_vectors[bkg_mask,...] = np.zeros([n, 2], dtype=np.float32)
    contribution_maps[bkg_mask,...] = np.ones([b, n], dtype=np.float32) * (1/n)

    src_map = np.zeros([h, w], dtype=np.uint8)
    src_map[source_locations[:, 0], source_locations[:, 1]] = 1

    # for each region
    # count sources and remove the extra sources
    segmentation_map, _ = label(src_mask)
    for region in regionprops(segmentation_map):
        start_x, start_y, stop_x, stop_y = region.bbox
        xs, ys = slice(start_x, stop_x), slice(start_y, stop_y)
        sub_segmap = region.image_filled
        n_srcs = src_map[ys, xs][sub_segmap].sum()

        # If there are at least as many sources as we can encode then we need to
        # keep all the claim vectors and claim maps
        # else less soures in the region than we are trying to encode, for
        # example a single source, so zero out the extra claim vectors/maps
        if n_srcs >= n:
            pass
        else:
            sub_cv = contribution_vectors[ys, xs, ...].copy() # [h', w', n, 2]
            sub_cm = contribution_maps[ys, xs, ...].copy()    # [h', w', b, n]

            sub_cv[background, n_srcs:, :] = 0

            sub_cm[background, :, n_srcs:] = 0
            sub_cm_totals = sub_cm.sum(axis=-1, keepdims=True) # [h', w', b, 1]
            sub_cm /= sub_cm_totals

            contribution_vectors[ys, xs, ...] = sub_cv
            contribution_maps[ys, xs, ...] = sub_cm

    return contribution_vectors, contribution_maps

def _encode_single_pixel(
    contribution_vectors: np.ndarray,
    contribution_maps: np.ndarray,
    scarlet_images: np.ndarray,
    source_locations: np.ndarray,
    n: int,
    y: int,
    x: int,
) -> None:
    """Encodes a single pixel into the PCR representation.
    shape labels:
        - h: height
        - w: width
        - b: bands
        - n: number of encoded sources
        - N: number of detected sources
    Args:
        contribution_vectors (np.ndarray): Shape [h, w, n, 2]. The contribution
                                           vectors for the entire image. This
                                           array is updated in-place
        contribution_maps (np.ndarray): Shape [h, w, b, n]. The contribution
                                        maps for the entire image. This array is
                                        updated in-place.
        scarlet_images (np.ndarray): Shape [N, b, h, w]. The SCARLET (Melchior
                                     et al., 2018) deblended source images.
        source_locations (np.ndarray): Shape [N, 2]. The source centers for each
                                       of the sources in `scarlet_images`
        n (int): The number of nearest sources to encode
        y (int): The y (h) location of the pixel to encode
        x (int): The x (w) location of the pixel to encode
    Returns:
        None, `contribution_vectors` & `contribution_maps` are updated in-place
    """

    # CONTRIBUTION VECTORS =====================================================
    relative_vectors = source_locations - np.array([y, x])  # [N, 2]
    relative_distances = np.linalg.norm(relative_vectors, axis=1)  # [N,]
    raw_closest_sources = np.argsort(relative_distances)[:n]  # [min(N, n),]

    # it's possible that there are less sources in the image than the number we
    # want to encode. In that case we need to pad the encoding with copies of
    # the last source.
    num_pad = n - raw_closest_sources.shape[0]
    if num_pad > 0:
        n_closest_sources = np.pad(
            raw_closest_sources, (0, num_pad), mode="edge"
        )  # [n,]
    else:
        n_closest_sources = raw_closest_sources  # [n,]

    vectors = relative_vectors[n_closest_sources, :]  # [n, 2]

    contribution_vectors[y, x, ...] = vectors
    # CONTRIBUTION VECTORS =====================================================

    # CONTRIBUTION MAPS ========================================================
    band_normed_flux = np.array(
        list(
            map(
                lambda b: _get_normed_src_flux(
                    scarlet_images, raw_closest_sources, b, y, x
                ),
                range(scarlet_images.shape[2]),
            )
        )
    )  # [b, min(N, n)]

    if num_pad > 0:
        padded_band_normed_flux = np.pad(
            band_normed_flux, ((0, 0), (0, num_pad)), mode="edge"
        )
    else:
        padded_band_normed_flux = band_normed_flux

    # If a source was duplicated to fill out the vector/map shapes, then we need
    # to reduce its contribution by the number of times it was duplicated
    idxs, counts = np.unique(
        n_closest_sources, return_counts=True
    )  # [min(n_srcs, n), ], [min(n_srcs, n), ]
    coefs = np.reciprocal(counts.astype(np.float32))  # [min(n_srcs, n), ]
    coef_map = np.array([coefs[idxs == i][0] for i in n_closest_sources])[
        np.newaxis, :
    ]  # [1, n]

    maps = padded_band_normed_flux * coef_map

    contribution_maps[y, x, ...] = maps
    # CONTRIBUTION MAPS ========================================================

    return None


def _get_normed_src_flux(
    scarlet_images: np.ndarray,
    source_idxs: np.ndarray,
    b: int,
    y: int,
    x: int,
) -> np.ndarray:
    """Gets the normalized flux for the location y,x,b for each source_idx
    Args:
        scarlet_images (np.ndarray): Shape [N, b, h, w]. The SCARLET (Melchior
                                     et al., 2018) deblended source images.
        source_idxs (np.ndarray): Shape [n]. The indicies of the sources
        b (int): The b (b) location of the pixel to encode
        y (int): The y (h) location of the pixel to encode
        x (int): The x (w) location of the pixel to encode
    Returns:
        np.darray of the normalized flux values for the each `source_idx` at the
        location (y,x,b)
    """
    source_fluxes = np.maximum(scarlet_images[source_idxs, b, y, x], 0)
    total_flux = source_fluxes.sum()

    if total_flux > 0:
        return source_fluxes / total_flux  # [n,]
    else:
        n = source_idxs.shape[0]
        return np.ones([n], dtype=np.float32) / n  # [n,]
