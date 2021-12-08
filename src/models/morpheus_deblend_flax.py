# MIT License
# Copyright 2021 Ryan Hausen
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

from functools import partial
from typing import Any, Callable, Tuple, Union

import gin
import jax.image as jim
import jax.numpy as jnp
import jax.random as rand
from flax import linen as nn

@gin.configurable()
class MorpheusDeblend(nn.Module):
    """A Detection/Deblending Model based on:

    Morpheus: A Deep Learning Framework for Pixel-Level Analysis of Astronomical
    Image Data https://arxiv.org/abs/1906.11248

    Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic
    Segmentation https://arxiv.org/abs/1911.10194

    Real-time Semantic Segmentation with Fast Attention
    https://arxiv.org/pdf/2007.03815.pdf

    Params:

    """

    pass


class FuseUp(nn.Module):
    """FuseUp Module used in decoders, combines skip connections and upsampling

    Based on FuseUp in "Real-time Semantic Segmentation with Fast Attention"

    Changed to allow non-matching dimension shapes between skip connections
    and upsampling inputs

    Args:
        filters (int): the number of filters to output
        activation (Callable): activation function default nn.relu
        resize_method (jax.image.ResizeMethod): Upsample interpolation method
                                                default linear
        dtype: dtype to output, default jnp.float32
    """
    filters: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    resize_method: jim.ResizeMethod = jim.ResizeMethod.LINEAR
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs:Tuple[jnp.ndarray, Union[jnp.ndarray, None]],
        train:bool
    ) -> jnp.ndarray:
        fuse_in, upsample_in = inputs

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            dtype=self.dtype,
        )

        conv = partial(
            nn.Conv,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=True,
            dtype=self.dtype,
        )


        if upsample_in is not None:
            upsample_in = jim.resize(
                upsample_in,
                [upsample_in.shape[0]*2, upsample_in.shape[1]*2, upsample_in.shape[2]],
                method=self.resize_method
            )
            upsample_in = self.activation(upsample_in)

            if upsample_in.shape[2] != fuse_in.shape[2]:
                fuse_in = conv(upsample_in.shape[2])(fuse_in)
                fuse_in = norm()(fuse_in)

            output = upsample_in + fuse_in
        else:
            output = fuse_in

        output = conv(self.filters)(output)
        return norm()(output)



class ResDown(nn.Module):
    filters:int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dtype:Any = jnp.float32

    def __call__(self, x:jnp.ndarray, train:bool) -> jnp.ndarray:

        conv = partial(
            nn.Conv,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=True,
            dtype=self.dtype,
        )



class AdaptiveFastAttenion(nn.Module):
    pass




def fuseup_tests():

    # Test no upsample input ===================================================
    x = jnp.zeros([100, 100, 4], dtype=jnp.float32)
    fuse_up = FuseUp(32)
    params = fuse_up.init(rand.PRNGKey(12171988), (x, None), True)

    y, bn_params = fuse_up.apply(params, (x, None), True, mutable=['batch_stats'])
    assert y.shape == (100, 100, 32)
    # Test no upsample input ===================================================

    # Test with upsample input =================================================
    x = jnp.zeros([100, 100, 4], dtype=jnp.float32)
    z = jnp.zeros([50, 50, 5], dtype=jnp.float32)
    fuse_up = FuseUp(32)
    params = fuse_up.init(rand.PRNGKey(12171988), (x, z), True)

    y, bn_params = fuse_up.apply(params, (x, z), True, mutable=['batch_stats'])
    assert y.shape == (100, 100, 32)
    # Test with upsample input =================================================

if __name__=="__main__":
    fuseup_tests()
