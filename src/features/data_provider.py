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
from typing import Tuple

import gin
import numpy as np
import tensorflow as tf
from astropy.io import fits

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")
DATA_PATH_PROCESSED = os.path.join(DATA_PATH, "processed")
TRAIN_DATA_PATH = os.path.join(DATA_PATH_PROCESSED, "train")
TEST_DATA_PATH = os.path.join(DATA_PATH_PROCESSED, "test")


def fits_load(fname):
    return fits.getdata(fname.numpy().decode("UTF-8"))


def open_fits(img_file):
    image = tf.py_function(fits_load, inp=[img_file], Tout=tf.float32)
    image.set_shape([512, 512, 9])
    return tf.data.Dataset.from_tensors(image)


def preprocess(sample):
    img = tf.slice(sample, [0, 0, 0], [512, 512, 4])
    lbl = tf.slice(sample, [0, 0, 4], [512, 512, 5])

    img = tf.image.random_crop(img, [512, 512, 1])

    return tf.concat([img, lbl], axis=-1)


@gin.configurable(blacklist=["sample"])
def augment(sample, jitter=True, flip_y=True, flip_x=True, normalize=True):
    tf.cast(sample, tf.float32)

    if jitter:
        height = width = int(512 * 1.117)
        sample = tf.image.random_crop(
            tf.image.resize(
                sample, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            ),
            size=[512, 512, 6],
        )
    if flip_y:
        sample = tf.image.random_flip_up_down(sample)
    if flip_x:
        sample = tf.image.random_flip_left_right(sample)
    if normalize:
        img = tf.slice(sample, [0, 0, 0], [512, 512, 1])
        lbl = tf.slice(sample, [0, 0, 1], [512, 512, 5])

        img = tf.image.per_image_standardization(img)
        sample = tf.concat([img, lbl], axis=-1)

    img = tf.slice(sample, [0, 0, 0], [512, 512, 1])
    lbl = tf.slice(sample, [0, 0, 1], [512, 512, 5])

    return sample


@gin.configurable
def get_dataset(
    batch_size: int,
) -> Tuple[Tuple[tf.data.Dataset, int], Tuple[tf.data.Dataset, int]]:
    options = tf.data.Options()
    options.experimental_optimization.map_vectorization.enabled = True

    train_files = tf.data.Dataset.list_files(TRAIN_DATA_PATH + "/*.fits", shuffle=False)
    num_train = len(os.listdir(TRAIN_DATA_PATH))
    train_batches_per_epoch = int(np.ceil(num_train / batch_size))

    dataset_train = (
        train_files.interleave(
            open_fits,
            cycle_length=batch_size,  # number of files to process in parallel
            block_length=1,  # how many items to produce from a single file
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .with_options(options)
        .repeat()
        .shuffle(batch_size + 1)
        .map(preprocess)
        .map(augment)
        .batch(batch_size)
    )

    test_files = tf.data.Dataset.list_files(TEST_DATA_PATH + "/*.fits", shuffle=False)
    # test_files = test_files.concatenate(
    #     tf.data.Dataset.from_tensor_slices(
    #         [os.path.join(DATA_PATH, "test_sample.fits")]
    #     )
    # )
    num_test = len(os.listdir(TEST_DATA_PATH)) + 1
    test_batches_per_epoch = int(np.ceil(num_test / batch_size))

    dataset_test = (
        test_files.interleave(
            open_fits,
            cycle_length=batch_size,  # number of files to process in parallel
            block_length=1,  # how many items to produce from a single file
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(preprocess)
        .batch(batch_size)
    )

    return (
        dataset_train,
        train_batches_per_epoch,
        dataset_test,
        test_batches_per_epoch,
    )
