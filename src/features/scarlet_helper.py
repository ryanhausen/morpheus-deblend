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
"""Handles SCARELET related functions."""

from functools import partial
from itertools import product, starmap
from typing import Callable, List, Tuple

import scarlet
import numpy as np

# A default function that gives an extended source prior
def all_extended_source(
    model_frame, observation, morpheus_label, source_yx,
) -> scarlet.component.FactorizedComponent:
    return scarlet.ExtendedSource(model_frame, source_yx, observation)


def get_scarlet_source(
    model_frame,
    observation,
    scarlet_sr_ctype_f: Callable,
    morpheus_label: np.ndarray,
    source_yx: Tuple[int, int],
) -> scarlet.component.FactorizedComponent:
    # For now go with the default scarlet recommendation of ExtendedSource with k=1
    return scarlet.ExtendedSource(model_frame, source_yx, observation)


def get_scarlet_fit(
    filters: List[str],
    psfs: np.ndarray,
    model_psf: Callable,
    flux: np.ndarray,  # [b, h, w]
    weights: np.ndarray,  # [b, h, w]
    source_locations: np.ndarray,  # [morphologies, h, w,]
) -> List[np.ndarray]:  # list of
    """Fit scarlet to image for generating labels"""

    model_frame = scarlet.Frame(flux.shape, psfs=model_psf, channels=filters)

    observation = scarlet.Observation(
        flux - flux.min(),
        psfs=scarlet.ImagePSF(psfs),
        weights=weights,
        channels=filters,
    ).match(model_frame)

    get_source = partial(
        get_scarlet_source,
        model_frame,
        observation,
        all_extended_source,
        source_locations,
    )

    src_ys, src_xs = np.nonzero(source_locations[0, :, :])
    sources = list(map(get_source, zip(src_ys, src_xs)))

    blend = scarlet.Blend(sources, observation)
    blend.fit(200, e_rel=1.0e-6)

    def render_source(obs, src):
        return obs.render(src.get_model(frame=src.frame))

    render_f = partial(render_source, observation)

    model_src_vals = list(map(render_f, sources))

    return model_src_vals
