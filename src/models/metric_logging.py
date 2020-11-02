
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

from itertools import starmap

import comet_ml
import gin
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean


import src.models.losses as losses


# TODO: These need be changed to weighted means
_semantic_loss = Mean()
_claim_vector_loss = Mean()
_claim_map_loss = Mean()
_center_of_mass_loss = Mean()
_total_loss = Mean()

@gin.configurable(whitelist=[
    "lambda_semantic",
    "lambda_claim_vector",
    "lambda_claim_map",
    "lambda_center_of_mass"
])
def update_metrics(
    experiment: comet_ml.Experiment,
    batches_per_epoch: int,
    is_training: bool,
    idx: int,
    inputs: np.ndarray,
    outputs: np.ndarray,
    lambda_semantic: float,
    lambda_claim_vector: float,
    lambda_claim_map: float,
    lambda_center_of_mass: float,
 ) -> None:

    epoch_progress = idx / batches_per_epoch

    flux, y_bkg, y_claim_vector, y_claim_map, y_com = inputs
    yh_bkg, yh_claim_vector, yh_claim_map, yh_com = outputs

    l_semantic = losses.semantic_loss(y=y_bkg, yh=yh_bkg)
    l_cv = losses.claim_vector_loss(bkg=y_bkg, y=y_claim_vector, yh=yh_claim_vector)
    l_cm = losses.claim_map_loss(bkg=y_bkg, y=y_claim_map, yh=yh_claim_map)
    l_com = losses.center_of_mass_loss(y=y_com, yh=yh_com)
    l_total = losses.loss_function(inputs, outputs)

    if is_training:
        metrics = [
            ("SemanticLoss", l_semantic * lambda_semantic),
            ("ClaimVectorLoss", l_cv * lambda_claim_vector),
            ("ClaimMapLoss", l_cm * lambda_claim_map),
            ("CenterOfMassLoss", l_com * lambda_center_of_mass),
            ("Loss", l_total),
        ]

        for _ in starmap(experiment.log_metric, metrics):
            pass
    else:

        _semantic_loss.update_state(l_semantic * lambda_semantic)
        _claim_vector_loss.update_state(l_cv * lambda_claim_vector)
        _claim_map_loss.update_state(l_cm * lambda_claim_map)
        _center_of_mass_loss.update_state(l_com * lambda_center_of_mass)
        _total_loss.update_state(l_total)

        if epoch_progress >= 1:
            metrics = [
                ("SemanticLoss", _semantic_loss),
                ("ClaimVectorLoss", _claim_vector_loss),
                ("ClaimMapLoss", _claim_map_loss),
                ("CenterOfMassLoss", _center_of_mass_loss),
                ("Loss", _total_loss),
            ]

            def send_and_reset(name, metric):
                experiment.log_metric(name, metric.result().numpy())
                metric.reset_states()

            for _ in starmap(send_and_reset, metrics):
                pass


