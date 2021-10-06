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
import os
from astropy.io import fits

from typing import Callable, Tuple, Union


import gin
import numpy as np
import tensorflow as tf

TensorLike = Union[tf.Tensor, np.ndarray]

LOCAL = os.path.dirname(__file__)
LOG_DIR = os.path.join(LOCAL, "data_log")

@gin.configurable(allowlist=[
    "lambda_claim_vector",
    "lambda_center_of_mass",
])
def spatial_loss_function(
    inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    outputs: Tuple[tf.Tensor, tf.Tensor],
    lambda_claim_vector: float,
    lambda_center_of_mass: float,
):
    flux, y_bkg, y_claim_vector, y_claim_map, y_com = inputs
    yh_claim_vector, yh_com = outputs

    loss = (
        lambda_claim_vector * claim_vector_loss(bkg=y_bkg, y_claim_map=y_claim_map, y=y_claim_vector, yh=yh_claim_vector, flux=flux)
        + lambda_center_of_mass * center_of_mass_loss(y=y_com, yh=yh_com, flux=flux)
    )

    return loss


@gin.configurable(allowlist=[
    "lambda_claim_map",
    "lambda_entropy_regularization",
])
def attribution_loss_function(
    inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    outputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    lambda_claim_map: float,
    lambda_entropy_regularization:float,
):

    flux, y_bkg, _, y_claim_map, _ = inputs
    yh_claim_map = outputs

    loss = (
        lambda_claim_map * claim_map_loss(bkg=y_bkg, y=y_claim_map, yh=yh_claim_map, flux=flux)
        + lambda_entropy_regularization * entropy_regularization(yh=yh_claim_map, flux=flux)
    )

    return loss



def nan_inf_detector(y, yh, loss, name):

    invalid_values_detected = tf.math.greater(
        tf.math.reduce_sum(
            tf.cast(
                tf.math.logical_not(
                    tf.math.is_finite(loss)
                ),
                tf.int32
            )
        ),
        tf.constant(0)
    )

    def raise_and_log():
        fits.PrimaryHDU(data=y.numpy()).writeto(
            os.path.join(LOG_DIR, f"{name}-y-val.fits"),
            overwrite=True
        )

        fits.PrimaryHDU(data=yh.numpy()).writeto(
            os.path.join(LOG_DIR, f"{name}-yh-val.fits"),
            overwrite=True
        )

        raise ValueError(f"{name}-Loss Raises a Nan/Inf")


    return tf.cond(invalid_values_detected, raise_and_log, lambda: loss)



@gin.configurable(allowlist=["loss_object", "avg"])
def semantic_loss(
    loss_object:tf.keras.losses.Loss, # use binary crossentropy
    avg:Callable,
    flux:TensorLike,    # [n, w, w, b]
    y:TensorLike,       # [n, h, w, 1]
    yh:TensorLike,      # [n, h, w, 1]
) -> float:

    loss = loss_object(y, yh) # [n, h, w]

    # any pixels with less than b are missing values and we want to exclude them
    augmentations_mask = tf.math.floordiv(
        tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(flux, 0), tf.float32),
            axis=-1,
        ),
        flux.shape[-1],
    )

    per_example_loss = tf.math.reduce_mean(loss * augmentations_mask, axis=(1, 2)) #[n,]

    #validated_loss = nan_inf_detector(y, yh, per_example_loss, "SemanticLoss")
    #return avg(validated_loss) # [0,]
    return avg(per_example_loss)

@gin.configurable(allowlist=["loss_object", "avg", "weighting"])
def claim_vector_loss(
    loss_object: tf.keras.losses.Loss, # use L1
    avg:Callable,
    weighting:str,
    flux:TensorLike,        # [n, h, w, b]
    bkg:TensorLike,         # [n, h, w, 1]
    y_claim_map:TensorLike, # [n, h, w, b, k]
    y:TensorLike,           # [n, h, w, k, 2]  k=8 for a neighborhood
    yh:TensorLike,          # [n, h, w, k, 2]
) -> float:

    per_k_loss = loss_object(y, yh) # [n, h, w, k]

    if weighting=="bkg":
        per_pixel_loss = tf.math.reduce_mean(per_k_loss, axis=-1) # [n, h, w]
        weighting = tf.math.abs(bkg[:, :, :, 0] - 1) # [n, h, w]
        per_pixel_loss = per_pixel_loss * weighting # [n, h, w]
    elif weighting=="claim_map":
        weighting = y_claim_map
        weighted_per_k_loss = tf.math.reduce_sum(per_k_loss * y_claim_map, axis=-1) # [n, h, w, b]
        per_pixel_loss = tf.math.reduce_mean(weighted_per_k_loss , axis=-1) # [n, h, w]
    elif weighting=="claim_map_and_bkg":
        weighting = tf.constant(1, dtype=tf.float32) - bkg[:, :, :, 0] # [n, h, w]
        weighted_per_k_loss = tf.math.reduce_sum(per_k_loss * y_claim_map, axis=-1) # [n, h, w, b]
        per_pixel_loss = tf.math.reduce_mean(weighted_per_k_loss, axis=-1) * weighting # [n, h, w]
    else:
        per_pixel_loss = tf.math.reduce_mean(per_k_loss, axis=-1) # [n, h, w]

    # any pixels with less than b are missing values and we want to exclude them
    # augmentations_mask = (flux != 0).sum(axis=-1) // flux.shape[-1] # [n, h, w]
    augmentations_mask = tf.math.floordiv(
        tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(flux, 0), tf.float32),
            axis=-1,
        ),
        flux.shape[-1],
    )

    per_example_loss = tf.math.reduce_mean(
        per_pixel_loss * augmentations_mask,
        axis=(1, 2)
    ) # [n,]

    return avg(per_example_loss)

    # validated_loss = nan_inf_detector(y, yh, per_example_loss, "ClaimVectorLoss")
    # return avg(validated_loss)

@gin.configurable(allowlist=["loss_object", "avg"])
def discrete_claim_vector_loss(
    loss_object: tf.keras.losses.Loss, # use L1, l2
    avg:Callable,
    bkg:TensorLike, # [n, h, w, 1]
    y:TensorLike, # [n, h, w, b, 8],
    yh:TensorLike, # [n, h, w, b, 8]
) -> float:

    weighting = tf.math.abs(bkg[:, :, :, 0] - 1) # [n, h, w]

    connected_loss = loss_object(y, yh) # [n, h, w, b]
    band_loss = tf.math.reduce_mean(connected_loss, axis=-1) # [n, h, w]

    per_pixel_loss = band_loss * weighting # [n, h, w]
    per_example_loss = tf.math.reduce_mean(per_pixel_loss, axis=(1, 2)) # [n,]
    return avg(per_example_loss)

@gin.configurable(allowlist=["loss_object", "avg", "weighting"])
def claim_map_loss(
    loss_object: tf.keras.losses.Loss, # use Categorical crossentrioy
    avg:Callable,
    bkg:TensorLike, # [n, h, w, 1]
    flux:TensorLike,  # [n, h, w, b]
    weighting:str,
    y:TensorLike, # [n, h, w, b, 8]
    yh:TensorLike # [n, h, w, b, 8]
) -> float:

    # any pixels with less than b are missing values and we want to exclude them
    # augmentations_mask = (flux != 0).sum(axis=-1) // flux.shape[-1] # [n, h, w]
    augmentations_mask = tf.math.floordiv(
        tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(flux, 0), tf.float32),
            axis=-1,
        ),
        flux.shape[-1],
    )

    band_loss = loss_object(y, yh)
    if weighting=="bkg":
        bkg_weighting = tf.math.abs(bkg[:, :, :, 0] - 1) # [n, h, w]

        weighting = augmentations_mask * bkg_weighting

        pixel_loss = tf.math.reduce_mean(band_loss, axis=-1) # [n, h, w]
        weighted_loss = pixel_loss * weighting # [n, h, w]
    else:
        pixel_loss = tf.math.reduce_mean(band_loss, axis=-1) # [n, h, w]
        weighted_loss = pixel_loss * augmentations_mask


    per_example_loss = tf.math.reduce_mean(weighted_loss, axis=(1, 2)) #[n,]
    return avg(per_example_loss)
    # validated_loss = nan_inf_detector(y, yh, per_example_loss, "ClaimMapLoss")
    # return avg(validated_loss) # [0,]


@gin.configurable(allowlist=["loss_object", "avg", "weighting"])
def center_of_mass_loss(
    loss_object: tf.keras.losses.Loss, # use L2 loss
    avg: Callable,
    weighting:str,
    flux:TensorLike,
    y:TensorLike, # [n, h, w, 1]
    yh:TensorLike
) -> float:
    # any pixels with less than b are missing values and we want to exclude them
    # augmentations_mask = (flux != 0).sum(axis=-1) // flux.shape[-1] # [n, h, w]
    augmentations_mask = tf.math.floordiv(
        tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(flux, 0), tf.float32),
            axis=-1,
        ),
        flux.shape[-1],
    )

    if weighting=="flux":
        loss = loss_object(y, yh) * augmentations_mask
    else:
        loss = loss_object(y, yh) * augmentations_mask

    per_example_loss = tf.math.reduce_mean(loss, axis=(1, 2)) # [n,]
    return avg(per_example_loss) # [0,]

@gin.configurable(allowlist=["avg", "weighting"])
def entropy_regularization(
    avg: Callable,
    weighting:str,
    flux:TensorLike,
    yh:TensorLike, # [n, h, w, b, k]
) -> float:

    # any pixels with less than b are missing values and we want to exclude them
    # augmentations_mask = (flux != 0).sum(axis=-1) // flux.shape[-1] # [n, h, w]
    augmentations_mask = tf.math.floordiv(
        tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(flux, 0), tf.float32),
            axis=-1,
        ),
        flux.shape[-1],
    )

    if weighting=="flux":
        pass
    else:
        yh_dist = tf.nn.softmax(yh) # [n, h, w, b, k]
        per_band_entropy = tf.math.reduce_mean(
            tf.math.xlogy(yh_dist, yh_dist) * -1,
            axis=-1
        ) # [n, h, w, b]
        per_pixel_entropy = tf.math.reduce_mean(per_band_entropy, axis=-1)

    per_example_entropy = tf.math.reduce_mean(
        per_pixel_entropy * augmentations_mask,
        axis=(1,2)
    )

    return avg(per_example_entropy)

@gin.configurable(allowlist=[
    "lambda_semantic",
    "lambda_claim_vector",
    "lambda_claim_map",
    "lambda_center_of_mass",
    "lambda_entropy_regularization",
    "instance_mode",
])
def loss_function(
    inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    outputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    lambda_semantic: float,
    lambda_claim_vector: float,
    lambda_claim_map: float,
    lambda_center_of_mass: float,
    lambda_entropy_regularization:float,
    instance_mode:str = "v1",
) -> float:

    if instance_mode in ["v1", "v4", "v5", "v6", "v7"]:
        flux, y_bkg, y_claim_vector, y_claim_map, y_com = inputs
        yh_bkg, yh_claim_vector, yh_claim_map, yh_com = outputs

        loss = (
            lambda_semantic * semantic_loss(y=y_bkg, yh=yh_bkg, flux=flux)
            + lambda_claim_vector * claim_vector_loss(bkg=y_bkg, y_claim_map=y_claim_map, y=y_claim_vector, yh=yh_claim_vector, flux=flux)
            + lambda_claim_map * claim_map_loss(bkg=y_bkg, y=y_claim_map, yh=yh_claim_map, flux=flux)
            + lambda_center_of_mass * center_of_mass_loss(y=y_com, yh=yh_com, flux=flux)
        )
    elif instance_mode=="v2":
        flux, y_bkg, y_claim_map, y_com = inputs
        yh_bkg, yh_claim_map, yh_com = outputs

        loss = (
            lambda_semantic * semantic_loss(y=y_bkg, yh=yh_bkg)
            + lambda_claim_map * claim_map_loss(bkg=y_bkg, y=y_claim_map, yh=yh_claim_map)
            + lambda_center_of_mass * center_of_mass_loss(y=y_com, yh=yh_com)
        )
    elif instance_mode=="v3":
        flux, y_bkg, y_claim_vector, y_claim_map, y_com = inputs
        yh_bkg, yh_claim_vector, yh_claim_map, yh_com = outputs

        loss = (
            lambda_semantic * semantic_loss(y=y_bkg, yh=yh_bkg)
            + lambda_claim_vector * discrete_claim_vector_loss(bkg=y_bkg, y=y_claim_vector, yh=yh_claim_vector)
            + lambda_claim_map * claim_map_loss(bkg=y_bkg, y=y_claim_map, yh=yh_claim_map)
            + lambda_center_of_mass * center_of_mass_loss(y=y_com, yh=yh_com)
        )
    elif instance_mode=="v8":
        flux, y_bkg, y_claim_vector, y_claim_map, y_com = inputs
        yh_claim_vector, yh_claim_map, yh_com = outputs

        loss = (
            lambda_claim_vector * claim_vector_loss(bkg=y_bkg, y_claim_map=y_claim_map, y=y_claim_vector, yh=yh_claim_vector, flux=flux)
            + lambda_claim_map * claim_map_loss(bkg=y_bkg, y=y_claim_map, yh=yh_claim_map, flux=flux)
            + lambda_center_of_mass * center_of_mass_loss(y=y_com, yh=yh_com, flux=flux)
            + lambda_entropy_regularization * entropy_regularization(yh=yh_claim_map, flux=flux)
        )
    else:
        raise ValueError("instance_mode must be equal to 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8'")


    return loss
