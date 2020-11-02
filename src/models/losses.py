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

from typing import Tuple, Union

import gin
import numpy as np
import tensorflow as tf

TensorLike = Union[tf.Tensor, np.ndarray]

@gin.configurable(whitelist=["loss_object"])
def semantic_loss(
    loss_object:tf.keras.losses.Loss, # use binary crossentropy
    y:TensorLike, # [n, h, w, 1]
    yh:TensorLike # [n, h, w, 1]
) -> float:

    loss = loss_object(y, yh) # [n, h, w]
    per_example_loss = tf.math.reduce_mean(loss, axis=(1, 2)) #[n,]
    return tf.nn.compute_average_loss(per_example_loss) # [0,]

@gin.configurable(whitelist=["loss_object"])
def claim_vector_loss(
    loss_object: tf.keras.losses.Loss, # use L1
    bkg:TensorLike, # [n, h, w, 1]
    y:TensorLike, # [n, h, w, b, 8, 2]
    yh:TensorLike, # [n, h, w, b, 8, 2]
) -> float:

    weighting = tf.math.abs(bkg[:, :, :, 0] - 1) # [n, h, w]

    connected_loss = loss_object(y, yh) # [n, h, w, 8]
    per_pixel_loss = tf.math.reduce_sum(connected_loss) * weighting # [n, h, w]
    pre_example_loss = tf.math.reduce_mean(per_pixel_loss, axis=(1, 2)) # [n,]
    return tf.nn.compute_average_loss(pre_example_loss)

@gin.configurable(whitelist=["loss_object"])
def claim_map_loss(
    loss_object: tf.keras.losses.Loss, # use Categorical crossentrioy
    bkg:TensorLike, # [n, h, w, 1]
    y:TensorLike, # [n, h, w, b, 8]
    yh:TensorLike # [n, h, w, b, 8]
) -> float:

    weighting = tf.math.abs(bkg[:, :, :, 0] - 1) # [n, h, w]

    band_loss = loss_object(y, yh) # [n, h, w, b]
    pixel_loss = tf.math.reduce_sum(band_loss, axis=-1) # [n, h, w]

    weighted_loss = pixel_loss * weighting # [n, h, w]
    per_example_loss = tf.math.reduce_mean(weighted_loss, axis=(1, 2)) #[n,]
    return tf.nn.compute_average_loss(per_example_loss) # [0,]


@gin.configurable(whitelist=["loss_object"])
def center_of_mass_loss(
    loss_object: tf.keras.losses.Loss, # use L2 loss
    y:TensorLike, # [n, h, w, 1]
    yh:TensorLike
) -> float:

    loss = loss_object(y, yh)
    per_example_loss = tf.math.reduce_mean(loss, axis=(1, 2)) # [n,]
    return tf.nn.compute_average_loss(per_example_loss) # [0,]


@gin.configurable(whitelist=[
    "lambda_semantic",
    "lambda_claim_vector",
    "lambda_claim_map",
    "lambda_center_of_mass"
])
def loss_function(
    inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    outputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    lambda_semantic: float,
    lambda_claim_vector: float,
    lambda_claim_map: float,
    lambda_center_of_mass: float,
) -> float:
    flux, y_bkg, y_claim_vector, y_claim_map, y_com = inputs
    yh_bkg, yh_claim_vector, yh_claim_map, yh_com = outputs


    return (
        lambda_semantic * semantic_loss(y=y_bkg, yh=yh_bkg)
        + lambda_claim_vector * claim_vector_loss(bkg=y_bkg, y=y_claim_vector, yh=yh_claim_vector)
        + lambda_claim_map * claim_map_loss(bkg=y_bkg, y=y_claim_map, yh=yh_claim_map)
        + lambda_center_of_mass * center_of_mass_loss(y=y_com, yh=yh_com)
    )