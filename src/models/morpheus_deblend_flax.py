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

import gin
import jax.numpy as jnp
from flax import linen as nn


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
    pass

class ResDown(nn.Module):
    pass

class AdaptiveFastAttenion(nn.Module):
    pass





