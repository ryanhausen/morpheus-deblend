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


def grab_files_py(parent_dir:str, instance_mode:str, idx:int):

    if instance_mode=="v1":
        fnames = ["flux", "background", "claim_vectors", "claim_maps", "center_of_mass"]
    elif instance_mode=="v2":
        fnames = ["flux", "background", "claim_maps", "center_of_mass"]
    elif instance_mode=="v3":
        fnames = ["flux", "background", "claim_vectors", "claim_maps", "center_of_mass"]
    elif instance_mode=="v4":
        fnames = ["flux", "background", "claim_vectors", "claim_maps", "center_of_mass"]
    elif instance_mode=="v5":
        fnames = ["flux", "background", "claim_vectors", "claim_maps", "center_of_mass"]
    else:
        raise ValueError("instance_mode must be one of ['v1', 'v2', 'v3', 'v4']")

    arrays = list(
        map(
            lambda suffix: fits.getdata(os.path.join(parent_dir, f"{idx.numpy().decode('UTF-8')}-{suffix}.fits")),
            fnames
        )
    )

    return arrays


def get_files(parent_dir:str, instance_mode:str, item_idx:tf.data.Dataset):

    if instance_mode=="v1":
        (
            flux,
            background,
            claim_vector,
            claim_map,
            center_of_mass
        ) = tf.py_function(
            partial(grab_files_py, parent_dir, instance_mode),
            inp=[item_idx],
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
        )

        flux.set_shape([256, 256, 4])
        background.set_shape([256, 256, 1])
        claim_vector.set_shape([256, 256, 4, 8, 2])
        claim_map.set_shape([256, 256, 4, 8])
        center_of_mass.set_shape([256, 256, 1])

        vals = (flux, background, claim_vector, claim_map, center_of_mass)
    elif instance_mode=="v2":
        n = 5

        (
            flux,
            background,
            claim_map,
            center_of_mass
        ) = tf.py_function(
            partial(grab_files_py, parent_dir, instance_mode),
            inp=[item_idx],
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32)
        )

        flux.set_shape([256, 256, 4])
        background.set_shape([256, 256, 1])
        claim_map.set_shape([256, 256, 4, n])
        center_of_mass.set_shape([256, 256, 1])

        vals = (flux, background, claim_map, center_of_mass)
    elif instance_mode=="v3":
        (
            flux,
            background,
            claim_vector,
            claim_map,
            center_of_mass
        ) = tf.py_function(
            partial(grab_files_py, parent_dir, instance_mode),
            inp=[item_idx],
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
        )

        flux.set_shape([256, 256, 4])
        background.set_shape([256, 256, 1])
        claim_vector.set_shape([256, 256, 4, 8])
        claim_map.set_shape([256, 256, 4, 8])
        center_of_mass.set_shape([256, 256, 1])

        vals = (flux, background, claim_vector, claim_map, center_of_mass)
    elif instance_mode in ["v4", "v5"]:
        n = 5
        (
            flux,
            background,
            claim_vector,
            claim_map,
            center_of_mass
        ) = tf.py_function(
            partial(grab_files_py, parent_dir, instance_mode),
            inp=[item_idx],
            Tout=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
        )

        flux.set_shape([256, 256, 4])
        background.set_shape([256, 256, 1])
        claim_vector.set_shape([256, 256, 4, n, 2])
        claim_map.set_shape([256, 256, 4, n])
        center_of_mass.set_shape([256, 256, 1])

        vals = (flux, background, claim_vector, claim_map, center_of_mass)
    else:
        raise ValueError(
            "instance_mode must be one of ['v1', 'v2', 'v3', 'v4', 'v5']"
        )

    return tf.data.Dataset.from_tensors(vals)


@gin.configurable()
def get_dataset(
    batch_size:int,
    instance_mode:str,
) -> Tuple[Tuple[tf.data.Dataset, int], Tuple[tf.data.Dataset, int]]:

    def get_idxs(directory:str):
        return list(set([f.split("-")[0] for f in os.listdir(directory)]))

    options = tf.data.Options()
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # Every training sample is composed of 5 files so divide the dir count by 5

    train_idxs = get_idxs(TRAIN_DATA_PATH)
    n_train = len(train_idxs)
    train_idxs = tf.data.Dataset.from_tensor_slices(train_idxs)

    # n_train = 5000
    # train_idxs = tf.data.Dataset.from_tensor_slices(
    #     ["6203"]
    # ).repeat(n_train)

    train_batches_per_epoch = int(np.ceil(n_train / batch_size))

    dataset_train = (
        train_idxs.interleave(
            partial(get_files, TRAIN_DATA_PATH, instance_mode),
            cycle_length=batch_size,  # number of files to process in parallel
            block_length=1,  # how many items to produce from a single file
            num_parallel_calls=2,
        )
        .with_options(options)
        .repeat()
        .shuffle(batch_size + 1)
        #.map(preprocess)
        #.map(augment)
        .batch(batch_size)
    )

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
            cycle_length=batch_size,  # number of files to process in parallel
            block_length=1,  # how many items to produce from a single file
            num_parallel_calls=2,
        )
        .with_options(options)
        .repeat()
        #.shuffle(batch_size + 1)
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
