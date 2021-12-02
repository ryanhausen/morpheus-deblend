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
"""Input function for the tf estimator API."""
import os
from functools import partial
from typing import Tuple

import gin
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from astropy.io import fits

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")
DATA_PATH_PROCESSED = os.path.join(DATA_PATH, "processed")
TRAIN_DATA_PATH = os.path.join(DATA_PATH_PROCESSED, "train")
TEST_DATA_PATH = os.path.join(DATA_PATH_PROCESSED, "test")


def open_file_py(parent_dir: str, suffix: str, idx: int):
    return fits.getdata(
        os.path.join(parent_dir, f"{idx.numpy().decode('UTF-8')}-{suffix}.fits")
    )


def grab_files_py(parent_dir: str, instance_mode: str, idx: int):

    if instance_mode == "v1":
        fnames = ["flux", "background", "claim_vectors", "claim_maps", "center_of_mass"]
    elif instance_mode == "v2":
        fnames = ["flux", "background", "claim_maps", "center_of_mass"]
    elif instance_mode == "v3":
        fnames = ["flux", "background", "claim_vectors", "claim_maps", "center_of_mass"]
    elif instance_mode == "v4":
        fnames = ["flux", "background", "claim_vectors", "claim_maps", "center_of_mass"]
    elif instance_mode == "v5":
        fnames = ["flux", "background", "claim_vectors", "claim_maps", "center_of_mass"]
    elif instance_mode in ["v6", "v7", "v8"]:
        fnames = ["flux", "background", "claim_vectors", "claim_maps", "center_of_mass"]
    else:
        raise ValueError(
            "instance_mode must be one of ['v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8']"
        )

    arrays = list(
        map(
            lambda suffix: fits.getdata(
                os.path.join(parent_dir, f"{idx.numpy().decode('UTF-8')}-{suffix}.fits")
            ),
            fnames,
        )
    )

    return arrays


def get_files(parent_dir: str, instance_mode: str, item_idx: tf.data.Dataset):

    if instance_mode == "v1":
        (flux, background, claim_vector, claim_map, center_of_mass) = tf.py_function(
            partial(grab_files_py, parent_dir, instance_mode),
            inp=[item_idx],
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        )

        flux.set_shape([256, 256, 4])
        background.set_shape([256, 256, 1])
        claim_vector.set_shape([256, 256, 4, 8, 2])
        claim_map.set_shape([256, 256, 4, 8])
        center_of_mass.set_shape([256, 256, 1])

        vals = (flux, background, claim_vector, claim_map, center_of_mass)
    elif instance_mode == "v2":
        n = 5

        (flux, background, claim_map, center_of_mass) = tf.py_function(
            partial(grab_files_py, parent_dir, instance_mode),
            inp=[item_idx],
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32),
        )

        flux.set_shape([256, 256, 4])
        background.set_shape([256, 256, 1])
        claim_map.set_shape([256, 256, 4, n])
        center_of_mass.set_shape([256, 256, 1])

        vals = (flux, background, claim_map, center_of_mass)
    elif instance_mode == "v3":
        (flux, background, claim_vector, claim_map, center_of_mass) = tf.py_function(
            partial(grab_files_py, parent_dir, instance_mode),
            inp=[item_idx],
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        )

        flux.set_shape([256, 256, 4])
        background.set_shape([256, 256, 1])
        claim_vector.set_shape([256, 256, 4, 8])
        claim_map.set_shape([256, 256, 4, 8])
        center_of_mass.set_shape([256, 256, 1])

        vals = (flux, background, claim_vector, claim_map, center_of_mass)
    elif instance_mode in ["v4", "v5", "v6"]:
        n = 3
        (flux, background, claim_vector, claim_map, center_of_mass) = tf.py_function(
            partial(grab_files_py, parent_dir, instance_mode),
            inp=[item_idx],
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        )

        flux.set_shape([256, 256, 4])
        background.set_shape([256, 256, 1])
        claim_vector.set_shape([256, 256, 4, n, 2])
        claim_map.set_shape([256, 256, 4, n])
        center_of_mass.set_shape([256, 256, 1])

        # vals = (
        #     flux,
        #     background,
        #     tf.reshape(claim_vector, [256, 256, -1]),
        #     tf.reshape(claim_map, [256, 256, -1]),
        #     center_of_mass
        # )

        # this HAS to be a tuple to not get stacked
        vals = (
            flux,
            background,
            claim_vector,
            claim_map,
            center_of_mass,
        )
    elif instance_mode in ["v7", "v8"]:
        n = 3
        n_bands = 1
        bands = slice(0, 1)
        (flux, background, claim_vector, claim_map, center_of_mass) = tf.py_function(
            partial(grab_files_py, parent_dir, instance_mode),
            inp=[item_idx],
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        )

        flux.set_shape([256, 256, 4])
        flux = flux[:, :, bands]

        background.set_shape([256, 256, 1])
        claim_vector.set_shape([256, 256, n, 2])
        claim_map.set_shape([256, 256, n_bands, n])
        center_of_mass.set_shape([256, 256, 1])

        vals = (
            flux,
            background,
            claim_vector,
            claim_map,
            center_of_mass,
        )
    else:
        raise ValueError(
            "instance_mode must be one of ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8']"
        )

    vals = tf.data.Dataset.from_tensors(vals)

    return vals


# @tf.function(experimental_relax_shapes=True)
def apply_and_shape(fn, inputs):
    shape = tf.shape(inputs)
    flat_channels = tf.reshape(inputs, [shape[0], shape[1], -1])

    return tf.reshape(fn(flat_channels), shape)


# @gin.configurable(allowlist=[
#    "flip_y",
#    "flip_x",
#    "translate_y",
#    "translate_x",
#    "rotate",
#    "instance_mode"
# ])
@tf.function
def augment(
    flux: tf.Tensor,
    background: tf.Tensor,
    claim_vector: tf.Tensor,
    claim_map: tf.Tensor,
    center_of_mass: tf.Tensor,
    flip_y: bool = True,
    flip_x: bool = True,
    translate_y: bool = True,
    translate_x: bool = True,
    rotate: bool = True,
    instance_mode: str = "v7",
):
    pi = tf.constant(np.pi, dtype=tf.float32)
    one_eighty = tf.constant(180, dtype=tf.float32)

    # Vertical Flip - 50/50
    if flip_y:
        if tf.math.greater(tf.random.uniform(shape=[]), 0.5):
            fn = partial(apply_and_shape, tf.image.flip_up_down)

            flux = fn(flux)
            background = fn(background)
            claim_vector = fn(claim_vector)
            claim_map = fn(claim_map)
            center_of_mass = fn(center_of_mass)

            tmp_y = claim_vector[..., 0] * -1
            tmp_x = claim_vector[..., 1]
            claim_vector = tf.stack([tmp_y, tmp_x], axis=-1)

    # Horizontal Flip - 50/50
    if flip_x:
        if tf.math.greater(tf.random.uniform(shape=[]), 0.5):
            fn = partial(apply_and_shape, tf.image.flip_left_right)

            flux = fn(flux)
            background = fn(background)
            claim_map = fn(claim_map)
            center_of_mass = fn(center_of_mass)

            claim_vector = fn(claim_vector)
            tmp_y = claim_vector[..., 0]
            tmp_x = claim_vector[..., 1] * -1
            claim_vector = tf.stack([tmp_y, tmp_x], axis=-1)

    # Vertical Shift - 50/50
    if translate_y:
        if tf.math.greater(tf.random.uniform(shape=[]), 0.5):
            dy = tf.random.uniform([], minval=0, maxval=25, dtype=tf.int32)
            fn = partial(
                apply_and_shape, partial(tfa.image.translate, translations=[dy, 0])
            )

            flux = fn(flux)
            background = fn(background)
            claim_vector = fn(claim_vector)
            claim_map = fn(claim_map)
            center_of_mass = fn(center_of_mass)

    # Horizontal Shift - 50/50
    if translate_x:
        if tf.math.greater(tf.random.uniform(shape=[]), 0.5):
            dx = tf.random.uniform([], minval=0, maxval=25, dtype=tf.int32)
            fn = partial(
                apply_and_shape, partial(tfa.image.translate, translations=[0, dx])
            )

            flux = fn(flux)
            background = fn(background)
            claim_vector = fn(claim_vector)
            claim_map = fn(claim_map)
            center_of_mass = fn(center_of_mass)

    # Rotate - 50/50
    if rotate:
        if tf.math.greater(tf.random.uniform(shape=[]), 0.5):
            theta = tf.random.uniform(shape=[], minval=1, maxval=360)
            rad = theta * pi / one_eighty

            fn = partial(apply_and_shape, partial(tfa.image.rotate, angles=rad),)

            flux = fn(flux)
            background = fn(background)
            claim_vector = fn(claim_vector)
            claim_map = fn(claim_map)
            center_of_mass = fn(center_of_mass)

            sin_theta = tf.sin(rad)
            cos_theta = tf.cos(rad)

            tmp_y = claim_vector[..., 0]
            tmp_x = claim_vector[..., 1]

            tmp_y = sin_theta * tmp_x + cos_theta * tmp_y
            tmp_x = cos_theta * tmp_x - sin_theta * tmp_y

            claim_vector = tf.stack([tmp_y, tmp_x], axis=-1)

    vals = (
        flux,
        background,
        claim_vector,
        claim_map,
        center_of_mass,
    )

    return vals


@gin.configurable()
def get_dataset(
    batch_size: int, instance_mode: str,
) -> Tuple[Tuple[tf.data.Dataset, int], Tuple[tf.data.Dataset, int]]:
    def get_idxs(directory: str):
        return list(set([f.split("-")[0] for f in os.listdir(directory)]))

    options = tf.data.Options()
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    # Every training sample is composed of 5 files so divide the dir count by 5

    train_idxs = get_idxs(TRAIN_DATA_PATH)
    n_train = len(train_idxs)
    train_idxs = tf.data.Dataset.from_tensor_slices(train_idxs)

    # n_train = 5000
    # train_idxs = tf.data.Dataset.from_tensor_slices(
    #     ["6203"]
    # ).repeat(n_train)

    train_batches_per_epoch = int(np.ceil(n_train / batch_size))

    # dataset_train = (
    #     train_idxs.interleave(
    #         partial(get_files, TRAIN_DATA_PATH, instance_mode),
    #         cycle_length=1,  # number of files to process in parallel
    #         block_length=1,  # how many items to produce from a single file
    #         num_parallel_calls=1,
    #     )
    #     # train_idxs.map(partial(get_files, TRAIN_DATA_PATH, instance_mode))
    #     # .with_options(options)
    #     .repeat()
    #     #.map(partial(augment, is_training=False))
    #     #.map(augment)
    #     .shuffle(batch_size + 1)
    #     #.map(preprocess)
    #     .batch(batch_size)
    # )

    dataset_train = train_idxs.interleave(
        partial(get_files, TRAIN_DATA_PATH, instance_mode),
        cycle_length=1,  # number of files to process in parallel
        block_length=1,  # how many items to produce from a single file
        num_parallel_calls=1,
    )
    # print("TRAINING: ", dataset_train)
    dataset_train = dataset_train.with_options(options)
    # print("TRAINING: ", dataset_train)
    dataset_train = dataset_train.repeat()
    # print("TRAINING: ", dataset_train)
    dataset_train = dataset_train.map(augment)
    # print("TRAINING: ", dataset_train)
    dataset_train = dataset_train.shuffle(batch_size + 1)
    # print("TRAINING: ", dataset_train)
    dataset_train = dataset_train.batch(batch_size)
    # print("TRAINING: ", dataset_train)

    test_idxs = get_idxs(TEST_DATA_PATH)
    n_test = len(test_idxs)
    test_idxs = tf.data.Dataset.from_tensor_slices(test_idxs)

    # n_test = batch_size * 10
    # test_idxs = tf.data.Dataset.from_tensor_slices(
    #     ["67"] # 67 is train item, put it back for actual dataset generation
    # ).repeat(n_test)

    test_batches_per_epoch = int(np.ceil(n_test / batch_size))

    dataset_test = (
        test_idxs.interleave(
            partial(get_files, TEST_DATA_PATH, instance_mode),
            cycle_length=1,  # number of files to process in parallel
            block_length=1,  # how many items to produce from a single file
            num_parallel_calls=1,
        )
        .with_options(options)
        .repeat()
        # .shuffle(batch_size + 1)
        # .map(preprocess)
        # .map(pure_augment)
        .batch(batch_size)
    )

    return (
        dataset_train,
        train_batches_per_epoch,
        dataset_test,
        test_batches_per_epoch,
    )


if __name__ == "__main__":
    dataset = get_dataset(1, "v7")

    (train, num_train, test, num_test) = dataset

    print("\n\n\n IN MAIN: ", train)

    for i, t in enumerate(train):
        print("\n\n in loop: ", t)

        for _i in t:
            print(_i.shape)

        break
