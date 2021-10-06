
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
import flow_vis
import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import expit, softmax
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

_semantic_loss = MeanOfMeans()
_claim_vector_loss = MeanOfMeans()
_claim_map_loss = MeanOfMeans()
_center_of_mass_loss = MeanOfMeans()
_total_loss = MeanOfMeans()
_entropy_regularization = MeanOfMeans()

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
    "lambda_entropy_regularization",
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
    lambda_entropy_regularization:float,
    instance_mode:str,
 ) -> None:
    n_instance = 3

    epoch_progress = idx / batches_per_epoch

    if instance_mode in ["v1", "v3", "v4", "v5", "v6", "v7"]:
        flux, y_bkg, y_claim_vector, y_claim_map, y_com = inputs
        yh_bkg, yh_claim_vector, yh_claim_map, yh_com = outputs
    elif instance_mode=="v2":
        flux, y_bkg, y_claim_map, y_com = inputs
        yh_bkg, yh_claim_map, yh_com = outputs
    elif instance_mode=="v8":
        flux, y_bkg, y_claim_vector, y_claim_map, y_com = inputs
        yh_claim_vector, yh_claim_map, yh_com = outputs
    elif instance_mode=="split":
        flux, y_bkg, y_claim_vector, y_claim_map, y_com = inputs
        yh_claim_vector, yh_com, yh_claim_map = outputs
    else:
        raise ValueError("instance_mode must in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8']")


    l_cm = losses.claim_map_loss(bkg=y_bkg, y=y_claim_map, yh=yh_claim_map, flux=flux)
    l_com = losses.center_of_mass_loss(y=y_com, yh=yh_com, flux=flux)
    l_entropy = losses.entropy_regularization(yh=yh_claim_map, flux=flux)
    l_total = losses.loss_function(inputs, outputs)

    if instance_mode not in  ["v8", "split"]:
        l_semantic = losses.semantic_loss(y=y_bkg, yh=yh_bkg, flux=flux)


    if instance_mode in ["v1", "v4", "v5", "v6", "v7", "v8", "split"]:
        l_cv = losses.claim_vector_loss(
            bkg=y_bkg,
            y_claim_map=y_claim_map,
            y=y_claim_vector,
            yh=yh_claim_vector,
            flux=flux,
        )
    if instance_mode=="v3":
        l_cv = losses.discrete_claim_vector_loss(
            bkg=y_bkg,
            y=y_claim_vector,
            yh=yh_claim_vector,
        )

    if is_training:
        metrics = [
            ("ClaimMapLoss", l_cm * lambda_claim_map),
            ("CenterOfMassLoss", l_com * lambda_center_of_mass),
            ("Loss", l_total),
            ("EntropyRegularization", l_entropy * lambda_entropy_regularization)
        ]

        if instance_mode not in ["v8", "split"]:
            metrics.append(
                ("SemanticLoss", l_semantic * lambda_semantic)
            )

        if instance_mode in ["v1", "v3", "v4", "v5", "v6", "v7", "v8", "split"]:
            metrics.append(
                ("ClaimVectorLoss", l_cv * lambda_claim_vector),
            )

        for _ in starmap(experiment.log_metric, metrics):
            pass


        if instance_mode != "v8":
            experiment.log_image(
                np.flipud(y_bkg[-1,...]),
                "InputBackground",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                np.flipud(expit(yh_bkg[-1,...])),
                "OutputBackground",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )


        experiment.log_image(
            np.flipud(y_com[-1,...]),
            "InputCenterOfMass",
            image_colormap="Greys",
            image_minmax=(0, 1)
        )

        experiment.log_image(
            np.flipud(yh_com[-1,...]),
            "OutputCenterOfMass",
            image_colormap="Greys",
            image_minmax=(0, 1)
        )

        experiment.log_image(
            np.flipud(scale_image(flux[-1,:, :, 0].numpy())),
            "Input-H",
            image_colormap="Greys"
        )

        # log claim map images
        if instance_mode=="v2":
            experiment.log_image(
                np.flipud(y_claim_map[-1, :, :, 0, 0]),
                "InputClaimMapClose1",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                np.flipud(yh_claim_map[-1, :, :, 0, 0]),
                "OutputClaimMapClose1",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                np.flipud(y_claim_map[-1, :, :, 0, 1]),
                "InputClaimMapClose2",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                np.flipud(yh_claim_map[-1, :, :, 0, 1]),
                "OutputClaimMapClose2",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

        # log vector images
        if instance_mode in ["v5", "v6", "v7", "v8", "split"]:

            # color vector representations
            cv_cm_vals = [
                (
                    y_claim_vector.numpy()[-1, ...],
                    y_claim_map.numpy()[-1, ...],
                ),
                (
                    yh_claim_vector.numpy()[-1, ...],
                    softmax(yh_claim_map.numpy()[-1, ...], axis=-1),
                ),
            ]

            names = ["Input", "Output"]

            for name, (cv, cm) in zip(names, cv_cm_vals):
                f, axes = plt.subplots(
                    ncols=2,
                    nrows=n_instance,
                    figsize=(8, 20),
                )

                for i, ax in enumerate(axes.flat):
                    single_cv = cv[:, :, i//2, :].copy()
                    single_cv[:, :, 0]  = single_cv[:, :, 0] * -1

                    # claim vector
                    if i % 2 == 0:
                        ax.imshow(
                            flow_vis.flow_to_color(
                                single_cv[..., [1, 0]],
                                convert_to_bgr=False
                            ),
                            origin="lower"
                        )
                    # claim map
                    else:
                        img_cmap = ax.imshow(
                            cm[:, :, 0, i//2],
                            vmin=0,
                            vmax=1,
                            cmap="magma",
                            origin="lower",
                        )
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        f.colorbar(img_cmap, cax=cax, orientation='vertical')

                    ax.set_xticks([])
                    ax.set_yticks([])

                    axes[0, 0].set_title("Claim Vectors")
                    axes[0, 1].set_title("Claim Maps")
                    plt.tight_layout()

                experiment.log_figure(
                    figure_name=f"{name}-CV/CM-Images",
                    figure=f,
                )
                plt.close(f)

            names = ["Input", "Output"]



            cv_y = y_claim_vector.numpy()[-1, ...] # [h, w, k, 2]
            cv_yh = yh_claim_vector.numpy()[-1, ...]   # [h, w, k, 2]

            f, axes = plt.subplots(
                ncols=2,
                nrows=n_instance,
                figsize=(8, 20),
            )

            for i, ax in enumerate(axes.flat):
                single_cv_y = cv_y[:, :, i//2, :].copy()   # [h, w, 2]
                single_cv_yh = cv_yh[:, :, i//2, :].copy() # [h, w, 2]

                mag_y = np.linalg.norm(single_cv_y, axis=-1)   # [h, w]
                mag_yh = np.linalg.norm(single_cv_yh, axis=-1) # [h, w]
                # cosine similarity
                if i % 2 == 0:

                    cos_sim = (single_cv_y * single_cv_yh).sum(axis=-1) / (mag_y * mag_yh)

                    img_cmap = ax.imshow(cos_sim, origin="lower", vmin=-1, vmax=1)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    f.colorbar(img_cmap, cax=cax, orientation='vertical')
                # magnitude difference
                else:
                    mag_diff = mag_y - mag_yh

                    img_cmap = ax.imshow(mag_diff, origin="lower")
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    f.colorbar(img_cmap, cax=cax, orientation='vertical')

                ax.set_xticks([])
                ax.set_yticks([])

            axes[0, 0].set_title("Cosine Similarity")
            axes[0, 1].set_title("Magnitude Difference")
            plt.tight_layout()

            experiment.log_figure(
                figure_name=f"{name}-CosSim/MagDiff",
                figure=f,
            )
            plt.close(f)


    else:
        n = flux.shape[0]
        idx = np.random.randint(n)


        _claim_map_loss.update_state(l_cm * lambda_claim_map, n)
        _center_of_mass_loss.update_state(l_com * lambda_center_of_mass, n)
        _total_loss.update_state(l_total, n)
        _entropy_regularization.update_state(l_entropy * lambda_entropy_regularization, n)


        if instance_mode not in  ["v8", "split"]:
            _semantic_loss.update_state(l_semantic * lambda_semantic, n)

        if instance_mode in ["v1", "v3", "v4", "v5", "v6", "v7", "v8", "split"]:
            _claim_vector_loss.update_state(l_cv * lambda_claim_vector, n)


        if epoch_progress >= 1:
            metrics = [
                ("ClaimMapLoss", _claim_map_loss),
                ("CenterOfMassLoss", _center_of_mass_loss),
                ("Loss", _total_loss),
                ("EntropyRegularization", _entropy_regularization)
            ]

            if instance_mode in ["v1", "v3", "v4", "v5", "v6", "v7", "v8", "split"]:
                metrics.append(
                    ("ClaimVectorLoss", _claim_vector_loss),
                )

            if instance_mode not in ["v8", "split"]:
                metrics.append(("SemanticLoss", _semantic_loss))

            def send_and_reset(name, metric):
                experiment.log_metric(name, metric.result().numpy())
                metric.reset_states()

            for _ in starmap(send_and_reset, metrics):
                pass


            if instance_mode not in ["v8", "split"]:
                experiment.log_image(
                    np.flipud(y_bkg[idx,...]),
                    "InputBackground",
                    image_colormap="Greys",
                    image_minmax=(0, 1)
                )

                experiment.log_image(
                    np.flipud(expit(yh_bkg[idx,...])),
                    "OutputBackground",
                    image_colormap="Greys",
                    image_minmax=(0, 1)
                )


            experiment.log_image(
                np.flipud(y_com[idx,...]),
                "InputCenterOfMass",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                np.flipud(yh_com[idx,...]),
                "OutputCenterOfMass",
                image_colormap="Greys",
                image_minmax=(0, 1)
            )

            experiment.log_image(
                np.flipud(scale_image(flux[idx,:, :, 0].numpy())),
                "Input-H",
                image_colormap="Greys"
            )

            if instance_mode=="v2":
                experiment.log_image(
                    np.flipud(y_claim_map[idx, :, :, 0, 0]),
                    "InputClaimMapClose1",
                    image_colormap="Greys",
                    image_minmax=(0, 1)
                )

                experiment.log_image(
                    np.flipud(yh_claim_map[idx, :, :, 0, 0]),
                    "OutputClaimMapClose1",
                    image_colormap="Greys",
                    image_minmax=(0, 1)
                )

                experiment.log_image(
                    np.flipud(y_claim_map[idx, :, :, 0, 1]),
                    "InputClaimMapClose2",
                    image_colormap="Greys",
                    image_minmax=(0, 1)
                )

                experiment.log_image(
                    np.flipud(yh_claim_map[idx, :, :, 0, 1]),
                    "OutputClaimMapClose2",
                    image_colormap="Greys",
                    image_minmax=(0, 1)
                )

            # log color vector representations
            if instance_mode in ["v5", "v6", "v7", "v8", "split"]:
                cv_cm_vals = [
                    (
                        y_claim_vector.numpy()[idx, ...],
                        y_claim_map.numpy()[idx, ...],
                    ),
                    (
                        yh_claim_vector.numpy()[idx, ...],
                        softmax(yh_claim_map.numpy()[idx, ...], axis=-1),
                    ),
                ]

                names = ["Input", "Output"]

                for name, (cv, cm) in zip(names, cv_cm_vals):
                    f, axes = plt.subplots(
                        ncols=2,
                        nrows=n_instance,
                        figsize=(8, 20),
                    )

                    for i, ax in enumerate(axes.flat):
                        single_cv = cv[:, :, i//2, :].copy()
                        single_cv[:, :, 0]  = single_cv[:, :, 0] * -1

                        # claim vector
                        if i % 2 == 0:
                            ax.imshow(
                                flow_vis.flow_to_color(
                                    single_cv[..., [1, 0]],
                                    convert_to_bgr=False
                                ),
                                origin="lower"
                            )
                        # claim map
                        else:
                            ax.imshow(
                                cm[:, :, 0, i//2],
                                vmin=0,
                                vmax=1,
                                cmap="magma",
                                origin="lower",
                            )
                        ax.set_xticks([])
                        ax.set_yticks([])

                        axes[0, 0].set_title("Claim Vectors")
                        axes[0, 1].set_title("Claim Maps")
                        plt.tight_layout()

                    experiment.log_figure(
                        figure_name=f"{name}-CV/CM-Images",
                        figure=f,
                    )
                    plt.close(f)

                cv_y = y_claim_vector.numpy()[idx, ...] # [h, w, k, 2]
                cv_yh = yh_claim_vector.numpy()[idx, ...]   # [h, w, k, 2]

                f, axes = plt.subplots(
                    ncols=2,
                    nrows=n_instance,
                    figsize=(8, 20),
                )

                for i, ax in enumerate(axes.flat):
                    single_cv_y = cv_y[:, :, i//2, :].copy()   # [h, w, 2]
                    single_cv_yh = cv_yh[:, :, i//2, :].copy() # [h, w, 2]

                    mag_y = np.linalg.norm(single_cv_y, axis=-1)   # [h, w]
                    mag_yh = np.linalg.norm(single_cv_yh, axis=-1) # [h, w]
                    # cosine similarity
                    if i % 2 == 0:

                        cos_sim = (single_cv_y * single_cv_yh).sum(axis=-1) / (mag_y * mag_yh)

                        img_cmap = ax.imshow(cos_sim, origin="lower", vmin=-1, vmax=1)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        f.colorbar(img_cmap, cax=cax, orientation='vertical')
                    # magnitude difference
                    else:
                        mag_diff = mag_y - mag_yh

                        img_cmap = ax.imshow(mag_diff, origin="lower")
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        f.colorbar(img_cmap, cax=cax, orientation='vertical')

                    ax.set_xticks([])
                    ax.set_yticks([])

                axes[0, 0].set_title("Cosine Similarity")
                axes[0, 1].set_title("Magnitude Difference")
                plt.tight_layout()

                experiment.log_figure(
                    figure_name=f"{name}-CosSim/MagDiff",
                    figure=f,
                )
                plt.close(f)
