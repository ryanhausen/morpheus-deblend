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
from functools import partial, reduce
from itertools import count, starmap
from typing import Callable, List, Tuple, Union

import gin
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model

LayerFunc = Callable[[tf.Tensor], tf.Tensor]
Tensorlike = Union[tf.Tensor, np.ndarray]

@gin.configurable()
def get_spatial_model(
    input_shape:List[str],
    encoder_filters:List[int],
    encoder_dropout_rate:float,
    output_shape:List[int],
    n_srcs:int,
    decoder_filters:List[int],
    decoder_dropout_rate:float,
) -> Model:

    inputs = layers.Input(shape=input_shape)

    encoded = encoder(
        input_shape=input_shape,
        filters=encoder_filters,
        dropout_rate=encoder_dropout_rate,
        name="MorpheusDeblendSpatialEncoder",
    )(inputs)

    outputs = spatial_decoder(
        output_shape=output_shape,
        filters=decoder_filters,
        n=n_srcs,
        dropout_rate=decoder_dropout_rate,
    )(list(reversed(encoded)))

    return Model([inputs], [outputs])

@gin.configurable()
def get_attribution_model(
    input_shape:List[str],
    encoder_filters:List[int],
    encoder_dropout_rate:float,
    output_shape:List[int],
    n_srcs:int,
    decoder_filters:List[int],
    decoder_dropout_rate:float,
) -> Model:

    inputs = layers.Input(shape=input_shape)

    encoded = encoder(
        input_shape=input_shape,
        filters=encoder_filters,
        dropout_rate=encoder_dropout_rate,
        name="MorpheusDeblendAttributionEncoder",
    )(inputs)

    outputs = attribution_decoder(
        output_shape=output_shape,
        filters=decoder_filters,
        n=n_srcs,
        dropout_rate=decoder_dropout_rate,
    )(list(reversed(encoded)))

    return Model([inputs], [outputs])


def encoder(
    input_shape: Tuple[int, int, int],
    filters: List[int],
    dropout_rate: float,
    name: str = "MorpheusDeblendEncoder",
) -> Model:
    """The encoder is a shared representation for both decoder modules.

    The encoder takes the input image as the input and returns the
    intermediate representations at each resolution.

    args:
        input_shape (List[int]): The shape of the input image ie [256,256,4]
        layers (List[int]): The number of filters in each block. The width
                            and height of the input will be reduced by
                            2*length(layers)
        name (str): Optional name to give to the model

    outputs:
        A list containing the final and intermediate representations from the
        encoder model.
    """

    name_prefixes = (f"ResNet_{i}" for i in count(1))
    res_funcs = starmap(
        lambda l, n: res_down(l, dropout_rate=dropout_rate, name_prefix=n),
        zip(filters, name_prefixes)
    )

    model_outputs = []

    def apply_res(x: tf.Tensor, func: LayerFunc) -> tf.Tensor:
        f_x = func(x)
        model_outputs.append(f_x)
        return f_x

    model_input = layers.Input(shape=input_shape, name=f"{name}_input")
    reduce(apply_res, res_funcs, model_input)

    return Model([model_input], model_outputs, name=name)


def attribution_decoder(
    output_shape: Tuple[int, int],
    filters: List[int],
    n: int,
    dropout_rate:float,
    name: str = "MorpheusDeblendAttributionDecoder",
) -> Model:
    hw = output_shape[0]
    input_shapes = list(
        reversed(
            [
                [hw // (2 ** i), hw // (2 ** i), c]
                for i, c in enumerate(filters, start=1)
            ]
        )
    )

    inputs = [layers.Input(shape=s) for s in input_shapes]

    attention_outs = list(starmap(
        lambda s, x: AdaptiveFastAttention(c_prime=s[2])(x),
        zip(input_shapes, inputs)
    ))

    fuse_funcs = list(
        map(
            lambda f: fuse_up(f, dropout_rate=dropout_rate, name_prefix=name),
            list(reversed(filters[:-1])) + [filters[-1]],
        )
    )
    funcs_inputs = list(zip(fuse_funcs, attention_outs))

    def apply_fuse(x: tf.Tensor, func_y: Tuple[LayerFunc, tf.Tensor]) -> tf.Tensor:
        func, y = func_y
        f_x = func(x, y)
        return f_x

    fuse_out = reduce(apply_fuse, funcs_inputs, None)
    up_out = layers.UpSampling2D()(fuse_out)
    pre_conv = layers.Conv2D(32, 5, padding="SAME")(up_out)
    claim_map_conv = layers.Conv2D(output_shape[2] * n, 1, padding="SAME")(pre_conv)
    claim_map = layers.Reshape(output_shape + [n])(claim_map_conv)

    return Model(inputs, [claim_map], name=name)


def spatial_decoder(
    output_shape: Tuple[int, int],
    filters: List[int],
    n: int,
    dropout_rate:float,
    name: str = "MorpheusDeblendSpatialDecoder",
) -> Model:
    hw = output_shape[0]
    input_shapes = list(
        reversed(
            [
                [hw // (2 ** i), hw // (2 ** i), c]
                for i, c in enumerate(filters, start=1)
            ]
        )
    )

    inputs = [layers.Input(shape=s) for s in input_shapes]

    attention_outs = list(starmap(
        lambda s, x: AdaptiveFastAttention(c_prime=s[2])(x),
        zip(input_shapes, inputs)
    ))

    fuse_funcs = list(
        map(
            lambda f: fuse_up(f, dropout_rate=dropout_rate, name_prefix=name),
            list(reversed(filters[:-1])) + [filters[-1]],
        )
    )
    funcs_inputs = list(zip(fuse_funcs, attention_outs))

    def apply_fuse(x: tf.Tensor, func_y: Tuple[LayerFunc, tf.Tensor]) -> tf.Tensor:
        func, y = func_y
        f_x = func(x, y)
        return f_x

    cv_fuse_out = reduce(apply_fuse, funcs_inputs, None)
    cv_up_out = layers.UpSampling2D()(cv_fuse_out)
    cv_pre_conv = partial(layers.Conv2D, 32, 5, padding="SAME")

    claim_vectors_conv = layers.Conv2D(
        output_shape[2] * n * 2,
        1,
        padding="SAME",
        name="CLAIM_VECTORS_CONV"
    )(cv_pre_conv()(cv_up_out))
    claim_vectors = layers.Reshape(output_shape[:-1] + [n, 2])(claim_vectors_conv)

    center_of_mass = layers.Conv2D(
        1, 1,
        padding="SAME",
        activation="sigmoid",
        name="CENTER_OF_MASS_CONV",
    )(cv_pre_conv()(cv_up_out))

    return Model(inputs, [claim_vectors, center_of_mass], name=name)


@gin.configurable(allowlist=["kernel_size", "activation"])
def res_down(
    filters: int,
    kernel_size: int = 3,
    activation: layers.Layer = layers.ReLU,
    dropout_rate:float = 0.5,
    name_prefix: str = "",
) -> LayerFunc:

    conv = partial(
        layers.Conv2D,
        filters,
        kernel_size,
        padding="SAME",
    )


    down_conv_f = lambda i, x: conv(strides=2, name=f"{name_prefix}_DCONV_{i}")(x)
    conv_f = lambda i, x: conv(name=f"{name_prefix}_CONV_{i}")(x)

    dropout_f = lambda i, x: layers.Dropout(rate=dropout_rate, name=f"{name_prefix}_DO_{i}")(x)

    bn_f = lambda i, x: layers.BatchNormalization(name=f"{name_prefix}_BN_{i}")(x)
    activation_f = lambda i, x: activation(name=f"{name_prefix}_ACTIVATION_{i}")(x)
    add_f = lambda i, x: layers.Add(name=f"{name_prefix}_ADD_{i}")(x)

    def block_1(x):
        one_a = "block1a_1"
        two_a = "block1a_2"

        # x => Conv2Ds2 => BatchNorm => Activation => Conv2D => BatchNorm
        dconv_a_out = down_conv_f(one_a, x)
        bn_a_out = bn_f(one_a, dconv_a_out)
        do_a_out = dropout_f(one_a, bn_a_out)
        act_a_out = activation_f(one_a, do_a_out)
        conv_a_out = conv_f(one_a, act_a_out)
        bn_a2_out = bn_f(two_a, conv_a_out)
        do_a2_out = dropout_f(two_a, bn_a2_out)

        one_b = "block1b_1"

        # x => DownConv2D => [_ + op_result] => Activation
        dconv_b_out = down_conv_f(one_b, x)
        add_out = add_f(one_b, [dconv_b_out, do_a2_out])
        act_b_out = activation_f(one_b, add_out)

        return act_b_out

    def block_2(x):
        one_a = "block2a_1"
        two_a = "block2a_2"

        # x => Conv2D => BatchNorm => Activation => Conv2D => BatchNorm
        conv_a_out = conv_f(one_a, x)
        bn_a_out = bn_f(one_a, conv_a_out)
        do_a_out = dropout_f(one_a, bn_a_out)
        act_a_out = activation_f(one_a, bn_a_out)
        conv_a2_out = conv_f(two_a, act_a_out)
        bn_a2_out = bn_f(two_a, conv_a2_out)
        do_a2_out = dropout_f(two_a, bn_a2_out)

        one_b = "block2b_1"

        # x => [_ + bn_a2_out] => Activation
        add_out = add_f(one_b, [do_a2_out, x])
        act_b_out = activation_f(one_b, add_out)

        return act_b_out

    return lambda x: block_2(block_1(x))


@gin.configurable(allowlist=["activation"])
def fuse_up(
    filters: int,
    activation: layers.Layer = layers.ReLU,
    dropout_rate:float = 0.5,
    name_prefix: str = "",
) -> LayerFunc:
    def op(x, y):
        if x is None:
            fused = y
        else:
            up_sampled = layers.UpSampling2D()(x)
            activated = activation()(up_sampled)
            dropped = layers.Dropout(dropout_rate)(activated)
            fused = layers.Add()([dropped, y])

        conv = layers.Conv2D(filters, (3, 3), padding="SAME")(fused)
        return layers.BatchNormalization()(conv)

    return lambda x, y: op(x, y)



class QKEncoder(tf.keras.layers.Layer):
    """Query and Key embedding layer"""

    def __init__(self, filters:int, **kwargs):
        super(QKEncoder, self).__init__(**kwargs)
        self.filters = filters
        self.conv = tf.keras.layers.Conv2D(filters, 1, padding="SAME")
        self.bn = tf.keras.layers.BatchNormalization()
        self.reshape = tf.keras.layers.Reshape([-1, filters])
        self.l2_norm = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=2))

    def call(self, inputs):
        return self.l2_norm(
            self.reshape(
                self.bn(
                    self.conv(
                        inputs
                    )
                )
            )
        )

    # Adapted from:
    # https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/layers/dense_attention.py#L483
    def get_config(self):
        config = dict(
            filters=self.filters,
        )
        base_config = super(QKEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VEncoder(tf.keras.layers.Layer):
    """Value embedding layer"""

    def __init__(self, filters:int, **kwargs):
        super(VEncoder, self).__init__(**kwargs)
        self.filters = filters
        self.conv = tf.keras.layers.Conv2D(filters, 1, padding="SAME")
        self.bn = tf.keras.layers.BatchNormalization()
        self.reshape = tf.keras.layers.Reshape([-1, filters])
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        bn = self.bn(self.conv(inputs))
        encoding = self.relu(self.reshape(bn))

        return encoding, bn

    # Adapted from:
    # https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/layers/dense_attention.py#L483
    def get_config(self):
        config = dict(
            filters=self.filters,
        )
        base_config = super(VEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



@gin.configurable(allowlist=["c_prime"])
class AdaptiveFastAttention(tf.keras.layers.Layer):
    """ Adaptive Fast Attention Layer.

    Based on:

    Real-time Semantic Segmentation with Fast Attention
    https://arxiv.org/pdf/2007.03815.pdf

    The change to the Fast Attention module is to vary the order of
    matrix multiplications operations according to the size of `n` and `c'`
    to minimize complexity.

    Args:
        c_prime (int): The number of attention features in q and k
    """

    def __init__(self,
        c_prime:int=128,
        out_filters:int=None,
        **kwargs
    ):
        super(AdaptiveFastAttention, self).__init__(**kwargs)
        self.c_prime = c_prime
        self.out_filters = out_filters

    @staticmethod
    def attention_qk_first(q, k, v):
        qk = tf.keras.layers.Dot(axes=(2, 2))([q, k])
        qkv = tf.keras.layers.Dot(axes=(2, 1))([qk, v])
        return qkv

    @staticmethod
    def attention_kv_first(q, k, v):
        kv = tf.keras.layers.Dot(axes=(1, 1))([k, v])
        qkv = tf.keras.layers.Dot(axes=(2, 1))([q, kv])
        return qkv


    def build(self, input_shape):
        h = w = input_shape[1]
        n = input_shape[1] * input_shape[2]
        c = input_shape[-1]
        out_c = self.out_filters if self.out_filters else c

        n_coef = 1 / n

        self.Q = QKEncoder(self.c_prime)
        self.K = QKEncoder(self.c_prime)
        self.V = VEncoder(c)

        qkv_cost = (n**2 * self.c_prime) + (n**2 * c)
        kvq_cost = (n * self.c_prime * c) * 2

        self.qk_first = qkv_cost < kvq_cost

        self.n_multiply = tf.keras.layers.Lambda(lambda x: n_coef * x)

        if self.qk_first:
            self.multiply_qkv = AdaptiveFastAttention.attention_qk_first
        else:
            self.multiply_qkv = AdaptiveFastAttention.attention_kv_first

        self.square_qkv = tf.keras.layers.Reshape([h, w, c])
        self.conv = tf.keras.layers.Conv2D(out_c, 3, padding="SAME")
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.residual_add = tf.keras.layers.Add()

    def call(self, inputs):
        q = self.Q(inputs)
        k = self.K(inputs)
        v, residual_v = self.V(inputs)

        qkv = self.n_multiply(self.multiply_qkv(q, k, v))

        out = self.relu(self.bn(self.conv(self.square_qkv(qkv))))

        return self.residual_add([residual_v, out])

    # Adapted from:
    # https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/layers/dense_attention.py#L483
    def get_config(self):
        config = dict(c_prime=self.c_prime)
        base_config = super(AdaptiveFastAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
