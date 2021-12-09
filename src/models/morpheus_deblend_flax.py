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
from typing import Any, Callable, List, Sequence, Tuple, Union
from warnings import filters
from flax.linen import activation

import gin
from jax._src.dtypes import dtype
from jax._src.lax.control_flow import X
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


class Encoder(nn.Module):
    filters: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dtype:Any = jnp.float32

    def __call__(self, x:jnp.ndarray, train:bool) -> List[jnp.ndarray]:

        outputs = []

        for f in self.filters:
            x = ResDown(f, self.activation, dtype=self.dtype)(x, train)
            outputs.append(x)

        return outputs


class Decoder(nn.Module):
    filters: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dtype:Any = jnp.float32

    def __call__(self, x:List[jnp.ndarray], train:bool) -> jnp.ndarray:

        upsample_input = None
        for attention_input, f in zip(x, self.filters):
            att_out = AdaptiveFastAttenion(
                activation=self.activation
            )(attention_input, train)
            upsample_input = FuseUp(
                f, activation=self.activation, dtype=self.dtype
            )(att_out, upsample_input, train)

        return upsample_input


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
        fuse_in:jnp.ndarray,
        upsample_in:Union[jnp.ndarray, None],
        train:bool
    ) -> jnp.ndarray:

        conv = partial(
            nn.Conv,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
        )

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
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


class ResidualBlock(nn.Module):
    filters: int
    downsample: bool
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x:jnp.ndarray, train:bool) -> jnp.ndarray:

        conv = partial(
            nn.Conv,
            self.filters,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
        )

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            dtype=self.dtype,
        )

        y = conv(strides=2**self.downsample)(x)
        y = norm()(y)
        y = self.activation(y)
        y = conv()(y)
        y = norm()(y)

        if self.downsample or x.shape[2] != self.filters:
            x = conv(
                strides=2**self.downsample,
                use_bias=True,
            )(x)

        x = x + y
        x = self.activation(x)
        return x



class ResDown(nn.Module):
    filters: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x:jnp.ndarray, train:bool) -> jnp.ndarray:

        x = ResidualBlock(self.filters, True, self.activation, self.dtype)(x, train)
        x = ResidualBlock(self.filters, False, self.activation, self.dtype)(x, train)

        return x


class QKEncoder(nn.Module):
    filters:int
    dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x:jnp.ndarray, train:bool) -> jnp.ndarray:

        conv = partial(
            nn.Conv,
            self.filters,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=True,
            dtype=self.dtype,
        )

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            dtype=self.dtype,
        )

        x = conv()(x)
        x = norm()(x)
        x = jnp.reshape(x, [-1, self.filters])
        x = x / jnp.linalg.norm(x, axis=0, keepdims=True, ord=2)
        return x


class VEncoder(nn.Module):
    filters:int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x:jnp.ndarray, train:bool) -> jnp.ndarray:
        conv = partial(
            nn.Conv,
            self.filters,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=True,
            dtype=self.dtype,
        )

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            dtype=self.dtype,
        )

        x = conv()(x)
        residual = norm()(x)
        x = jnp.reshape(residual, [-1, self.filters])
        x = self.activation(x)
        return x, residual


class AdaptiveFastAttenion(nn.Module):
    qk_embedding_dim:int = None
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dtype=Any = jnp.float32

    @nn.compact
    def __call__(self, x:jnp.ndarray, train:bool) -> jnp.ndarray:
        h, w, c = x.shape
        n = h * w

        if self.qk_embedding_dim is None:
            self.qk_embedding_dim = c

        q = QKEncoder(self.qk_embedding_dim)(x, train)
        k = QKEncoder(self.qk_embedding_dim)(x, train).T
        v, residual_v = VEncoder(c, activation=self.activation)(x, train)

        qkv_cost = (n**2 * self.qk_embedding_dim) + (n**2 * c)
        kvq_cost = (n * self.qk_embedding_dim * c) * 2

        if qkv_cost > kvq_cost:
            x = (1/n) * jnp.dot(q, jnp.dot(k, v))
        else:
            x = (1/n) * jnp.dot(jnp.dot(q, k), v)

        x = jnp.reshape(x, [h, w, c])
        x = nn.Conv(
            c,
            kernel_size=(3, 3),
            use_bias=False,
            padding="SAME",
            dtype=jnp.float32
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            dtype=jnp.float32
        )(x)

        x = self.activation(x)

        return x + residual_v


def fuseup_tests():

    # Test no upsample input ===================================================
    x = jnp.zeros([100, 100, 4], dtype=jnp.float32)
    fuse_up = FuseUp(32)
    params = fuse_up.init(rand.PRNGKey(12171988), x, None, True)

    y, bn_params = fuse_up.apply(params, x, None, True, mutable=["batch_stats"])
    assert y.shape == (100, 100, 32)
    # Test no upsample input ===================================================

    # Test with upsample input =================================================
    x = jnp.zeros([100, 100, 4], dtype=jnp.float32)
    z = jnp.zeros([50, 50, 5], dtype=jnp.float32)
    fuse_up = FuseUp(32)
    params = fuse_up.init(rand.PRNGKey(12171988), x, z, True)

    y, bn_params = fuse_up.apply(params, x, z, True, mutable=["batch_stats"])
    assert y.shape == (100, 100, 32)
    # Test with upsample input =================================================

def residual_block_tests():

    # test downsample ==========================================================
    x = jnp.zeros([100, 100, 4], dtype=jnp.float32)
    resblock = ResidualBlock(32, True)
    params = resblock.init(rand.PRNGKey(12171988), x, True)

    y, bn_params = resblock.apply(params, x, True, mutable=["batch_stats"])
    assert y.shape ==(50, 50, 32)
    # test downsample ==========================================================

    # test no downsample =======================================================
    x = jnp.zeros([100, 100, 32], dtype=jnp.float32)
    resblock = ResidualBlock(32, False)
    params = resblock.init(rand.PRNGKey(12171988), x, True)

    y, bn_params = resblock.apply(params, x, True, mutable=["batch_stats"])
    assert y.shape ==(100, 100, 32)
    # test no downsample =======================================================

def resdown_tests():

    x = jnp.zeros([100, 100, 4], dtype=jnp.float32)
    resblock = ResDown(32)
    params = resblock.init(rand.PRNGKey(12171988), x, True)

    y, bn_params = resblock.apply(params, x, True, mutable=["batch_stats"])
    assert y.shape == (50, 50, 32)

def qkencoder_tests():
    x = jnp.ones([100, 100, 4], dtype=jnp.float32)
    qkencoder = QKEncoder(32)
    params = qkencoder.init(rand.PRNGKey(12171988), x, False)

    y, bn_params = qkencoder.apply(params, x, False, mutable=["batch_stats"])

    assert y.shape == (100 * 100, 32)
    assert jnp.allclose(jnp.ones([32], dtype=jnp.float32), jnp.linalg.norm(y, axis=0))

def vencoder_tests():
    x = jnp.ones([100, 100, 32], dtype=jnp.float32)
    vencoder = VEncoder(32)
    params = vencoder.init(rand.PRNGKey(12171988), x, False)

    (y, residual_y), bn_params = vencoder.apply(params, x, False, mutable=["batch_stats"])

    assert y.shape == (100 * 100, 32)

def adaptivefastattention_tests():
    x = jnp.ones([100, 100, 32], dtype=jnp.float32)
    afa = AdaptiveFastAttenion()
    params = afa.init(rand.PRNGKey(12171988), x, False)

    y, bn_params = afa.apply(params, x, False, mutable=["batch_stats"])

    assert y.shape == (100, 100, 32)

if __name__=="__main__":
    fuseup_tests()
    residual_block_tests()
    resdown_tests()
    qkencoder_tests()
    vencoder_tests()
    adaptivefastattention_tests()