
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
from scipy.special import expit
from tensorflow.keras.metrics import Mean


import src.models.losses as losses


class MeanOfMeans(tf.keras.metrics.Metric):

    def __init__(self, name="mean_of_means", **kwargs):
        super(MeanOfMeans, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(name="x", initializer="zeros")
        self.n = self.add_weight(name="n", initializer="zeros")

    def update_state(self, x, n):
        self.sum.assign_add(x * n)
        self.n.assign_add(n)

    def result(self):
        return self.sum / self.n

# TODO: These need be changed to weighted means
_semantic_loss = MeanOfMeans()
_claim_vector_loss = MeanOfMeans()
_claim_map_loss = MeanOfMeans()
_center_of_mass_loss = MeanOfMeans()
_total_loss = MeanOfMeans()


def scale_image(data):

    ox = int(data.shape[0]/4)
    oy = int(data.shape[1]/4)
    nx = int(data.shape[0]/2)
    ny = int(data.shape[1]/2)

    s = np.std(data[ox:ox+nx,oy:oy+ny])
    m = np.mean(data[ox:ox+nx,oy:oy+ny])

    ret = (data-m)/s
    ret = np.log10((data-m)/s + 1.0e-6 - (data-m).min()/s)
    m = np.mean(ret)
    s = np.std(ret)

    ret[ret<m-0.5*s] = m-0.5*s
    ret[ret>m+2.0] = m+2.0
    ret = (ret - ret.min())/(ret.max()-ret.min())
    return ret

@gin.configurable(allowlist=[
    "lambda_semantic",
    "lambda_claim_vector",
    "lambda_claim_map",
    "lambda_center_of_mass",
    "instance_mode",
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
    instance_mode:str,
 ) -> None:

    epoch_progress = idx / batches_per_epoch



    if instance_mode in ["v1", "v3", "v4", "v5"]:
        flux, y_bkg, y_claim_vector, y_claim_map, y_com = inputs
        yh_bkg, yh_claim_vector, yh_claim_map, yh_com = outputs
    elif instance_mode=="v2":
        flux, y_bkg, y_claim_map, y_com = inputs
        yh_bkg, yh_claim_map, yh_com = outputs
    else:
        raise ValueError("instance_mode must in ['v1', 'v2', 'v3', 'v4', 'v5']")

    l_semantic = losses.semantic_loss(y=y_bkg, yh=yh_bkg)
    l_cm = losses.claim_map_loss(bkg=y_bkg, y=y_claim_map, yh=yh_claim_map)
    l_com = losses.center_of_mass_loss(y=y_com, yh=yh_com)
    l_total = losses.loss_function(inputs, outputs)

    if instance_mode in ["v1", "v4", "v5"]:
        l_cv = losses.claim_vector_loss(
            bkg=y_bkg,
            y=y_claim_vector,
            yh=yh_claim_vector
        )
    if instance_mode=="v3":
        l_cv = losses.discrete_claim_vector_loss(
            bkg=y_bkg,
            y=y_claim_vector,
            yh=yh_claim_vector
        )

    if is_training:
        metrics = [
            ("SemanticLoss", l_semantic * lambda_semantic),
            ("ClaimMapLoss", l_cm * lambda_claim_map),
            ("CenterOfMassLoss", l_com * lambda_center_of_mass),
            ("Loss", l_total),
        ]

        if instance_mode in ["v1", "v3", "v4", "v5"]:
            metrics.append(
                ("ClaimVectorLoss", l_cv * lambda_claim_vector),
            )

        for _ in starmap(experiment.log_metric, metrics):
            pass

        experiment.log_image(
            y_bkg[-1,...],
            "InputBackground",
            image_colormap="Greys",
            image_minmax=(0, 1)
        )

        experiment.log_image(
            expit(yh_bkg[-1,...]),
            "OutputBackground",
            image_colormap="Greys",
            image_minmax=(0, 1)
        )


        experiment.log_image(
            y_com[-1,...],
            "InputCenterOfMass",
            image_colormap="Greys",
            image_minmax=(0, 1)
        )

        experiment.log_image(
            yh_com[-1,...],
            "OutputCenterOfMass",
            image_colormap="Greys",
            image_minmax=(0, 1)
        )

        experiment.log_image(
            scale_image(flux[-1,:, :, 0].numpy()),
            "Input-H",
            image_colormap="Greys"
        )

        if instance_mode=="v2":
            experiment.log_image(
                y_claim_map[-1, :, :, 0, 0],
                "InputClaimMapClose1",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                yh_claim_map[-1, :, :, 0, 0],
                "OutputClaimMapClose1",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                y_claim_map[-1, :, :, 0, 1],
                "InputClaimMapClose2",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                yh_claim_map[-1, :, :, 0, 1],
                "OutputClaimMapClose2",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

    else:
        n = flux.shape[0]
        _semantic_loss.update_state(l_semantic * lambda_semantic, n)
        _claim_map_loss.update_state(l_cm * lambda_claim_map, n)
        _center_of_mass_loss.update_state(l_com * lambda_center_of_mass, n)
        _total_loss.update_state(l_total, n)

        if instance_mode in ["v1", "v3", "v4", "v5"]:
            _claim_vector_loss.update_state(l_cv * lambda_claim_vector, n)


        if epoch_progress >= 1:
            metrics = [
                ("SemanticLoss", _semantic_loss),
                ("ClaimMapLoss", _claim_map_loss),
                ("CenterOfMassLoss", _center_of_mass_loss),
                ("Loss", _total_loss),
            ]

            if instance_mode in ["v1", "v3", "v4", "v5"]:
                metrics.append(
                    ("ClaimVectorLoss", _claim_vector_loss),
                )

            def send_and_reset(name, metric):
                experiment.log_metric(name, metric.result().numpy())
                metric.reset_states()

            for _ in starmap(send_and_reset, metrics):
                pass


            experiment.log_image(
                y_bkg[-1,...],
                "InputBackground",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                expit(yh_bkg[-1,...]),
                "OutputBackground",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )


            experiment.log_image(
                y_com[-1,...],
                "InputCenterOfMass",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                yh_com[-1,...],
                "OutputCenterOfMass",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                scale_image(flux[-1,:, :, 0].numpy()),
                "Input-H",
                image_colormap="Greys"
            )

            if instance_mode=="v2":
                experiment.log_image(
                    y_claim_map[-1, :, :, 0, 0],
                    "InputClaimMapClose1",
                    image_colormap="Greys",
                    image_minmax=(0, 1)
                )

                experiment.log_image(
                    yh_claim_map[-1, :, :, 0, 0],
                    "OutputClaimMapClose1",
                    image_colormap="Greys",
                    image_minmax=(0, 1)
                )

                experiment.log_image(
                    y_claim_map[-1, :, :, 0, 1],
                    "InputClaimMapClose2",
                    image_colormap="Greys",
                    image_minmax=(0, 1)
                )

                experiment.log_image(
                    yh_claim_map[-1, :, :, 0, 1],
                    "OutputClaimMapClose2",
                    image_colormap="Greys",
                    image_minmax=(0, 1)
                )
