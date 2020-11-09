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
from astropy.io import fits

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")
DATA_PATH_PROCESSED = os.path.join(DATA_PATH, "processed")
TRAIN_DATA_PATH = os.path.join(DATA_PATH_PROCESSED, "train")
TEST_DATA_PATH = os.path.join(DATA_PATH_PROCESSED, "test")


def grab_files_py(parent_dir:str, idx:int):
    #i = idx.numpy().decode("UTF-8")

    arrays = list(
        map(
            lambda suffix: fits.getdata(os.path.join(parent_dir, f"{idx.numpy().decode('UTF-8')}-{suffix}.fits")),
            ["flux", "background", "claim_vectors", "claim_maps", "center_of_mass"]
        )
    )

    return arrays


def get_files(parent_dir:str, item_idx:tf.data.Dataset):

    flux, background, claim_vector, claim_map, center_of_mass = tf.py_function(
        partial(grab_files_py, parent_dir),
        inp=[item_idx],
        Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
    )

    flux.set_shape([256, 256, 4])
    background.set_shape([256, 256, 1])
    claim_vector.set_shape([256, 256, 4, 8, 2])
    claim_map.set_shape([256, 256, 4, 8])
    center_of_mass.set_shape([256, 256, 1])


    return tf.data.Dataset.from_tensors((
        flux,
        background,
        claim_vector,
        claim_map,
        center_of_mass,
    ))


@gin.configurable()
def get_dataset(
    batch_size:int
) -> Tuple[Tuple[tf.data.Dataset, int], Tuple[tf.data.Dataset, int]]:

    options = tf.data.Options()
    options.experimental_optimization.map_vectorization.enabled = True

    # Every training sample is composed of 5 files so divide the dir count by 5

    # n_train = len(os.listdir(TRAIN_DATA_PATH)) // 5
    # train_idxs = tf.data.Dataset.from_tensor_slices(tf.range(n_train))

    n_train = 5000
    train_idxs = tf.data.Dataset.from_tensor_slices(
        ["6203"]
    ).repeat(n_train)

    train_batches_per_epoch = int(np.ceil(n_train / batch_size))

    dataset_train = (
        train_idxs.interleave(
            partial(get_files, TRAIN_DATA_PATH),
            cycle_length=batch_size,  # number of files to process in parallel
            block_length=1,  # how many items to produce from a single file
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .with_options(options)
        .repeat()
        .shuffle(batch_size + 1)
        #.map(preprocess)
        #.map(augment)
        .batch(batch_size)
    )


    #n_test = len(os.listdir(TEST_DATA_PATH)) // 5
    #test_idxs = tf.data.Dataset.from_tensor_slices(tf.range(n_test))


    n_test = batch_size * 10
    test_idxs = tf.data.Dataset.from_tensor_slices(
        ["67"] # 67 is train item, put it back for actual dataset generation
    ).repeat(n_test)

    test_batches_per_epoch = int(np.ceil(n_test / batch_size))

    dataset_test = (
        test_idxs.interleave(
            partial(get_files, TEST_DATA_PATH),
            cycle_length=batch_size,  # number of files to process in parallel
            block_length=1,  # how many items to produce from a single file
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .with_options(options)
        .repeat()
        .shuffle(batch_size + 1)
        #.map(preprocess)
        #.map(augment)
        .batch(batch_size)
    )


    return (
        dataset_train,
        train_batches_per_epoch,
        dataset_test,
        test_batches_per_epoch
    )


if __name__=="__main__":
    dataset = get_dataset(1)

    (train, num_train, test, num_test) = dataset

    for i, t in enumerate(train):
        for _i in t:
            print(_i.shape)