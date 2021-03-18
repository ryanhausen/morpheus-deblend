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

import argparse
from functools import partial
import os
import time
import psutil

from itertools import starmap
from pathlib import Path
from typing import Callable, Tuple, Union


# if we are on lux pynvml is python 2 in
# /cm/local/apps/cuda/libs/418.67/pynvml/pynvml.py
if os.getenv("ON_SLURM", default=None):
    import sys

    sys.path.reverse()


import comet_ml
import gin
import gin.tf
import numpy as np
import tensorflow as tf
from tqdm import tqdm


from src.features.data_provider import get_dataset
from src.models.utils import config_str_to_dict

LOCAL = Path(__file__).parent.absolute()


@gin.configurable()
def training_func(
    model: tf.keras.models.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    metric_func: Callable,
    checkpoint_dir: str,
    epochs: int,
    log_metric_batch_idx: int,
    model_code_file: str,
    comet_disabled: bool,
    experiment_project_name: str,
    experiment_key: str,
    strategy: tf.distribute.Strategy = None,
) -> None:
    checkpoint_dir = os.path.join(LOCAL, checkpoint_dir)

    if experiment_key:
        experiment = comet_ml.ExistingExperiment(
            #api_key=os.getenv("comet_key"),
            previous_experiment=experiment_key
        )
    else:
        experiment = comet_ml.Experiment(
            os.getenv("comet_key"),
            project_name=experiment_project_name,
            disabled=comet_disabled,
            auto_metric_logging=False,
        )
        experiment.log_parameters(config_str_to_dict(gin.config_str()))

        print("experiment: ", experiment)
        print("LOCAL: ", LOCAL)
        print("model_code_file: ", model_code_file)
        print("os.path.join: ", os.path.join)
        # experiment.log_code(
        #     file_name=os.path.join(LOCAL, model_code_file)
        # )
        experiment_key = experiment.get_key()

    # TODO: This should be stored in figured out somehow
    if strategy:
        with strategy.scope():
            model = model()
            optimizer = optimizer()
            experiment_step = tf.Variable(1)
            checkpoint = tf.train.Checkpoint(
                model=model, optimizer=optimizer, experiment_step=experiment_step,
            )
            checkpoint_manager = tf.train.CheckpointManager(
                checkpoint=checkpoint,
                directory=os.path.join(checkpoint_dir, experiment_key),
                max_to_keep=3,
            )
        n_workers = strategy.num_replicas_in_sync
    else:
        model = model()
        optimizer = optimizer()
        experiment_step = tf.Variable(1)
        checkpoint = tf.train.Checkpoint(
            model=model, optimizer=optimizer, experiment_step=experiment_step,
        )
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=os.path.join(checkpoint_dir, experiment_key),
            max_to_keep=3,
        )
        n_workers = 1

    experiment_step = tf.Variable(1)
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, experiment_step=experiment_step,
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=os.path.join(checkpoint_dir, experiment_key),
        max_to_keep=3,
    )

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print(f"Training from {checkpoint_manager.latest_checkpoint}")
    else:
        print("Training from scratch")

    # the params for get_dataset are set by gin
    (
        training_data,
        train_batches_per_epoch,
        testing_data,
        test_batches_per_epoch,
    ) = get_dataset()  # pylint: disable=no-value-for-parameter

    if strategy:
        training_data = strategy.experimental_distribute_dataset(training_data)
        testing_data = strategy.experimental_distribute_dataset(testing_data)

    # Training Functions vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    train_step = partial(step, strategy, model, True, optimizer=optimizer)
    if strategy:
        train_step_f = train_step
    else:
        train_step_f = train_step

    train_post_step_f = partial(
        post_step,
        experiment,
        experiment_step,
        strategy,
        checkpoint_manager.save,
        log_metric_batch_idx,
        True,
        partial(metric_func, experiment, train_batches_per_epoch, True)
    )
    train_epoch_finished = lambda idx: idx == train_batches_per_epoch
    # Training Functions ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Testing Functions vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    test_step = partial(step, strategy, model, False)
    if strategy:
        test_step_f = test_step
    else:
        test_step_f = test_step

    test_post_step_f = partial(
        post_step,
        experiment,
        experiment_step,
        strategy,
        lambda :None,
        log_metric_batch_idx, # is ignored when train==False
        False,
        partial(metric_func, experiment, test_batches_per_epoch, False)
    )
    test_epoch_finished = lambda idx: idx == test_batches_per_epoch
    # Testing Functions ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


    def epoch_f(epoch: int) -> None:
        print("Epoch: ", epoch)
        experiment.set_epoch(epoch)
        start_time = time.time()

        # Training Steps vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        train_batch_idx = starmap(
            train_post_step_f,
            starmap(
                train_step_f,
                enumerate(training_data, start=1)
            )
        )

        next(filter(
            train_epoch_finished,
            tqdm(train_batch_idx, total=train_batches_per_epoch-1, desc="Training")
        ))
        # Training Steps ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


        # Testing Steps vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        test_batch_idx = starmap(
            test_post_step_f,
            starmap(
                test_step_f,
                enumerate(testing_data, start=1)
            )
        )

        next(filter(
            test_epoch_finished,
            tqdm(test_batch_idx, total=test_batches_per_epoch-1, desc="Testing")
        ))
        # Testing Steps ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        print(f"\nEpoch completed in {np.round(time.time()-start_time, 2)} seconds")


    for _ in map(epoch_f, range(epochs)):
        with open("/home/rhausen/jobs/morpheus-deblend/use_stats.log", "a") as f:
            f.write(f"Memory: {psutil.virtual_memory().used / 1024 / 1024 / 1024} GB" + "\n")
        pass


@tf.function
def execute_step(
    model:tf.keras.models.Model,
    inputs:Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
    ],  # flux, bkg, claim_v, claim_m, com
    loss_func: Callable,
    is_training:bool,
    optimizer: tf.keras.optimizers.Optimizer = None
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    flux = inputs[0]

    if is_training:
        with tf.GradientTape() as tape:
            outputs = model(flux, training=is_training)
            loss = loss_func(inputs=inputs, outputs=outputs)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    else:
        outputs = model(flux, training=is_training)

    return outputs

@gin.configurable(allowlist=["loss_func"])
def step(
    strategy:tf.distribute.Strategy,
    model: tf.keras.models.Model,
    is_training: bool,
    batch_idx: int,
    inputs: Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
    ],  # flux, bkg, claim_v, claim_m, com
    loss_func: Callable = None,
    optimizer: tf.keras.optimizers.Optimizer = None,
) -> Tuple[
    int,
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],  # inputs
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor],  # model outputs: bkg, claim_v, claim_m, com
]:
    if strategy:
        outputs = strategy.run(execute_step, args=(model, inputs, loss_func, is_training, optimizer))
    else:
        outputs = execute_step(model, inputs, loss_func, is_training, optimizer)

    return (batch_idx, inputs, outputs)


def post_step(
    experiment,
    experiment_step: tf.Variable,
    strategy: tf.distribute.Strategy,
    save_f:Callable,
    log_batch_idx:int,
    is_training: bool,
    metric_func: Callable,
    batch_idx: int,
    inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    outputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
) -> int:

    if is_training:
        context = experiment.train
        experiment_step.assign_add(1)
        should_log = experiment_step.numpy() % log_batch_idx == 0
    else:
        context = experiment.test
        should_log = True

    if should_log:

        if strategy:
            inputs = [tf.concat(i.values, axis=0) for i in inputs]
            outputs = [tf.concat(o.values, axis=0) for o in outputs]


        experiment.set_step(experiment_step.numpy())
        with context():
            metric_func(
                batch_idx,
                inputs,
                outputs
            )

            save_f()

    return batch_idx


def main(config_file: str):
    strategy = tf.distribute.MirroredStrategy()
    strategy = None

    gin.config.external_configurable(tf.keras.losses.BinaryCrossentropy, module='tf.keras.losses')
    gin.config.external_configurable(tf.keras.losses.CategoricalCrossentropy, module='tf.keras.losses')
    gin.config.external_configurable(tf.keras.losses.MeanAbsoluteError, module='tf.keras.losses')
    gin.config.external_configurable(tf.keras.losses.MeanSquaredError, module='tf.keras.losses')
    gin.config.external_configurable(tf.nn.compute_average_loss, "tf.nn.compute_average_loss")

    gin.parse_config_file(os.path.join(LOCAL, config_file))
    training_func(strategy=strategy)  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Trainer")
    parser.add_argument("config", help="Gin config file with model params.")
    main(parser.parse_args().config)
