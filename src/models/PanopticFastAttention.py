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
"""Implements A Panoptic Version of the FastAttention Network for Morpheus.

Panoptic-DeepLab:
A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation
https://arxiv.org/abs/1911.10194


Real-time Semantic Segmentation with Fast Attention
https://arxiv.org/pdf/2007.03815.pdf
"""

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

# ==============================================================================
# Instance Modes:
#
# v1: neigborhood claim vectors are assigned to the source nearest to them.
#     claim maps and center of mass are like PanopticDeepLab
#
# v2: claim vectors are no longer used, claim maps and com are unchanged
#
# v3: discete claim vectors, model predicts magnitude
# ==============================================================================


@gin.configurable()
def get_model(input_shape = List[int], instance_mode:str = "v1") -> Model:

    inputs = layers.Input(shape=input_shape)

    enc = encoder()
    dec_semantic = semantic_decoder()

    if instance_mode=="v1":
        dec_instance = instance_decoder()
    elif instance_mode=="v2":
        dec_instance = instance_decoder_v2()
    elif instance_mode=="v3":
        dec_instance = instance_decoder_v3()
    else:
        raise ValueError("Instance mode should be 'v1', 'v2', or 'v3'")

    enc_outputs = enc(inputs)
    reversed_outputs = list(reversed(enc_outputs))

    bkg = dec_semantic(reversed_outputs)
    instance = dec_instance(reversed_outputs)

    return Model([inputs], [bkg] + instance)

@gin.configurable()
def encoder(
    input_shape: Tuple[int, int, int],
    filters: List[int],
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
        lambda l, n: res_down(l, name_prefix=n), zip(filters, name_prefixes)
    )

    model_outputs = []

    def apply_res(x: tf.Tensor, func: LayerFunc) -> tf.Tensor:
        f_x = func(x)
        model_outputs.append(f_x)
        return f_x

    model_input = layers.Input(shape=input_shape, name=f"{name}_input")
    reduce(apply_res, res_funcs, model_input)

    return Model([model_input], model_outputs, name=name)


@gin.configurable()
def semantic_decoder(
    output_shape: Tuple[int, int],
    filters: List[int],
    n_classes: int,
    name: str = "MorpheusDeblendSemanticDecoder",
) -> Model:
    """The semantic decoder module outputs a class map for semantic segmentation.

    """
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
    att_outs = list(starmap(
        lambda s, x: AdaptiveFastAttention(s[2])(x),
        zip(input_shapes, inputs)
    ))

    # We map the filters in reverse order because we start small and grow the
    # output back to the input resolution. We add an additional filter for the
    # final upsample and out
    fuse_funcs = list(
        map(
            lambda f: fuse_up(f, name_prefix=name),
            list(reversed(filters[:-1])) + [filters[-1]],
        )
    )
    fuse_funcs_y = list(zip(fuse_funcs, att_outs))

    def apply_fuse(x: tf.Tensor, func_y: Tuple[LayerFunc, tf.Tensor]) -> tf.Tensor:
        func, y = func_y
        f_x = func(x, y)
        return f_x

    fuse_out = reduce(apply_fuse, fuse_funcs_y, None)
    up_out = layers.UpSampling2D()(fuse_out)
    conv_out = layers.Conv2D(256, 5, padding="SAME")(up_out)
    final_conv_out = layers.Conv2D(n_classes, 1)(conv_out)

    return Model(inputs, [final_conv_out], name=name)


@gin.configurable()
def instance_decoder(
    output_shape: Tuple[int, int],
    filters: List[int],
    name: str = "MorpheusDeblendInstanceDecoder",
) -> Model:
    """The instance decoder outputs the values needed for source separation.

    """
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
    att_outs = list(map(lambda x: AdaptiveFastAttention()(x), inputs))

    # We map the filters in reverse order because we start small and grow the
    # output back to the input resolution. We add an additional filter for the
    # final upsample and out
    fuse_funcs = list(
        map(
            lambda f: fuse_up(f, name_prefix=name),
            list(reversed(filters[:-1])) + [filters[-1]],
        )
    )
    fuse_funcs_y = list(zip(fuse_funcs, att_outs))

    def apply_fuse(x: tf.Tensor, func_y: Tuple[LayerFunc, tf.Tensor]) -> tf.Tensor:
        func, y = func_y
        f_x = func(x, y)
        return f_x

    fuse_out = reduce(apply_fuse, fuse_funcs_y, None)
    up_out = layers.UpSampling2D()(fuse_out)

    pre_conv = partial(layers.Conv2D, 32, 5, padding="SAME")

    claim_vectors_conv = layers.Conv2D(output_shape[2] * 16, 1, padding="SAME")(pre_conv()(up_out))
    claim_vectors = layers.Reshape(output_shape + [8, 2])(claim_vectors_conv)

    claim_map_conv = layers.Conv2D(output_shape[2] * 8, 1, padding="SAME")(pre_conv()(up_out))
    claim_map = layers.Reshape(output_shape + [8])(claim_map_conv)

    center_of_mass = layers.Conv2D(
        1, 1,
        padding="SAME",
        activation="sigmoid"
        )(pre_conv()(up_out))

    return Model(inputs, [claim_vectors, claim_map, center_of_mass], name=name)

@gin.configurable()
def instance_decoder_v2(
    output_shape: Tuple[int, int],
    filters: List[int],
    n:int,
    name:str = "MorpheusDeblendInstanceDecoder"
) -> Model:
    """The instance decoder outputs the values needed for source separation.

    """
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
    att_outs = list(map(lambda x: AdaptiveFastAttention()(x), inputs))

    # We map the filters in reverse order because we start small and grow the
    # output back to the input resolution. We add an additional filter for the
    # final upsample and out
    fuse_funcs = list(
        map(
            lambda f: fuse_up(f, name_prefix=name),
            list(reversed(filters[:-1])) + [filters[-1]],
        )
    )
    fuse_funcs_y = list(zip(fuse_funcs, att_outs))

    def apply_fuse(x: tf.Tensor, func_y: Tuple[LayerFunc, tf.Tensor]) -> tf.Tensor:
        func, y = func_y
        f_x = func(x, y)
        return f_x

    fuse_out = reduce(apply_fuse, fuse_funcs_y, None)
    up_out = layers.UpSampling2D()(fuse_out)

    pre_conv = partial(layers.Conv2D, 32, 5, padding="SAME")

    claim_map_conv = layers.Conv2D(output_shape[2] * n, 1, padding="SAME")(pre_conv()(up_out))
    claim_map = layers.Reshape(output_shape + [n])(claim_map_conv)

    center_of_mass = layers.Conv2D(
        1, 1,
        padding="SAME",
        activation="sigmoid"
        )(pre_conv()(up_out))

    return Model(inputs, [claim_map, center_of_mass], name=name)

@gin.configurable()
def instance_decoder_v3(
    output_shape: Tuple[int, int],
    filters: List[int],
    name:str ="MorpheusDeblendInstanceDecoder",
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
    att_outs = list(starmap(
        lambda s, x: AdaptiveFastAttention(c_prime=s[2])(x),
        zip(input_shapes, inputs)
    ))

    # We map the filters in reverse order because we start small and grow the
    # output back to the input resolution. We add an additional filter for the
    # final upsample and out
    fuse_funcs = list(
        map(
            lambda f: fuse_up(f, name_prefix=name),
            list(reversed(filters[:-1])) + [filters[-1]],
        )
    )
    fuse_funcs_y = list(zip(fuse_funcs, att_outs))

    def apply_fuse(x: tf.Tensor, func_y: Tuple[LayerFunc, tf.Tensor]) -> tf.Tensor:
        func, y = func_y
        f_x = func(x, y)
        return f_x

    fuse_out = reduce(apply_fuse, fuse_funcs_y, None)
    up_out = layers.UpSampling2D()(fuse_out)

    pre_conv = partial(layers.Conv2D, 32, 5, padding="SAME")

    claim_vectors_conv = layers.Conv2D(output_shape[2] * 8, 1, padding="SAME")(pre_conv()(up_out))
    claim_vectors = layers.Reshape(output_shape + [8])(claim_vectors_conv)

    claim_map_conv = layers.Conv2D(output_shape[2] * 8, 1, padding="SAME")(pre_conv()(up_out))
    claim_map = layers.Reshape(output_shape + [8])(claim_map_conv)

    center_of_mass = layers.Conv2D(
        1, 1,
        padding="SAME",
        activation="sigmoid"
    )(pre_conv()(up_out))

    return Model(inputs, [claim_vectors, claim_map, center_of_mass], name=name)

@gin.configurable(allowlist=["kernel_size", "activation"])
def res_down(
    filters: int,
    kernel_size: int = 3,
    activation: layers.Layer = layers.ReLU,
    name_prefix: str = "",
) -> LayerFunc:

    conv = partial(layers.Conv2D, filters, kernel_size, padding="SAME")
    down_conv_f = lambda i, x: conv(strides=2, name=f"{name_prefix}_DCONV_{i}")(x)
    conv_f = lambda i, x: conv(name=f"{name_prefix}_CONV_{i}")(x)

    bn_f = lambda i, x: layers.BatchNormalization(name=f"{name_prefix}_BN_{i}")(x)
    activation_f = lambda i, x: activation(name=f"{name_prefix}_ACT_{i}")(x)
    add_f = lambda i, x: layers.Add(name=f"{name_prefix}_ADD_{i}")(x)

    def block_1(x):
        one_a = "block1a_1"
        two_a = "block1a_2"

        # x => Conv2Ds2 => BatchNorm => Activation => Conv2D => BatchNorm
        dconv_a_out = down_conv_f(one_a, x)
        bn_a_out = bn_f(one_a, dconv_a_out)
        act_a_out = activation_f(one_a, bn_a_out)
        conv_a_out = conv_f(one_a, act_a_out)
        bn_a2_out = bn_f(two_a, conv_a_out)

        one_b = "block1b_1"

        # x => DownConv2D => [_ + op_result] => Activation
        dconv_b_out = down_conv_f(one_b, x)
        add_out = add_f(one_b, [dconv_b_out, bn_a2_out])
        act_b_out = activation_f(one_b, add_out)

        return act_b_out

    def block_2(x):
        one_a = "block2a_1"
        two_a = "block2a_2"

        # x => Conv2D => BatchNorm => Activation => Conv2D => BatchNorm
        conv_a_out = conv_f(one_a, x)
        bn_a_out = bn_f(one_a, conv_a_out)
        act_a_out = activation_f(one_a, bn_a_out)
        conv_a2_out = conv_f(two_a, act_a_out)
        bn_a2_out = bn_f(two_a, conv_a2_out)

        one_b = "block2b_1"

        # x => [_ + bn_a2_out] => Activation
        add_out = add_f(one_b, [bn_a2_out, x])
        act_b_out = activation_f(one_b, add_out)

        return act_b_out

    return lambda x: block_2(block_1(x))


@gin.configurable(allowlist=["activation"])
def fuse_up(
    filters: int, activation: layers.Layer = layers.ReLU, name_prefix: str = "",
) -> LayerFunc:
    def op(x, y):
        if x is None:
            fused = y
        else:
            up_sampled = layers.UpSampling2D()(x)
            activated = activation()(up_sampled)
            fused = layers.Add()([activated, y])

        conv = layers.Conv2D(filters, (3, 3), padding="SAME")(fused)
        return layers.BatchNormalization()(conv)

    return lambda x, y: op(x, y)



class QKVEncoder(tf.keras.layers.Layer):
    """Quey, Key, and Value embedding"""

    def __init__(self, filters:int, QKV:str, **kwargs):
        super(QKVEncoder, self).__init__(**kwargs)
        self.filters = filters
        self.conv = layers.Conv2D(filters, 1)
        self.bn = layers.BatchNormalization()
        self.reshape = layers.Reshape([-1, filters])
        self.qkv = QKV

        if QKV in ["Q", "K"]:
            self.final_op = lambda x: K.l2_normalize(x, axis=2)
        elif QKV in ["V"]:
            self.final_op = lambda x: K.relu(x)
        else:
            raise ValueError("QKC must be one of 'Q', 'K', 'V'")


    def call(self, inputs):
        bn = self.bn(self.conv(inputs))
        encoding = self.final_op(self.reshape(bn))

        if self.qkv in ["Q", "K"]:
            return encoding
        else:
            return (encoding, bn)


    # Adapted from:
    # https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/layers/dense_attention.py#L483
    def get_config(self):
        config = dict(c_prime=self.c_prime)
        base_config = super(QKVEncoder, self).get_config()
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

    def __init__(self, c_prime:int=128, **kwargs):
        super(AdaptiveFastAttention, self).__init__(**kwargs)
        self.c_prime = c_prime


    @staticmethod
    def attention_qk_first(q, k, v):
        qk = layers.Dot(axes=(2, 2))([q, k])
        qkv = layers.Dot(axes=(2, 1))([qk, v])
        return qkv

    @staticmethod
    def attention_kv_first(q, k, v):
        kv = layers.Dot(axes=(1, 1))([k, v])
        qkv = layers.Dot(axes=(2, 1))([q, kv])
        return qkv


    def build(self, input_shape):
        batch_size, h, w, c = input_shape
        n = h * w

        self.n_coef = tf.constant(
            1 / n,
            dtype=tf.float32
        )

        self.Q = QKVEncoder(self.c_prime, "Q")
        self.K = QKVEncoder(self.c_prime, "K")
        self.V = QKVEncoder(c, "V")

        qk_v_cost = (n**2 * self.c_prime) + (n**2 * c)
        q_kv_cost = (n * self.c_prime * c) * 2

        self.qk_first = qk_v_cost < q_kv_cost

        if self.qk_first:
            self.multiply_qkv = AdaptiveFastAttention.attention_qk_first
        else:
            self.multiply_qkv = AdaptiveFastAttention.attention_kv_first

        self.square_qkv = tf.keras.layers.Reshape([h, w, c])

        self.conv = layers.Conv2D(c, 3, padding="SAME")
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

        self.residual_add = layers.Add()

    def call(self, inputs):
        q = self.Q(inputs)
        k = self.K(inputs)
        v, residual_v = self.V(inputs)

        qkv = self.multiply_qkv(q, k, v)

        out = self.relu(self.bn(self.conv(self.square_qkv(qkv))))

        return self.residual_add([residual_v, out])

    # Adapted from:
    # https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/layers/dense_attention.py#L483
    def get_config(self):
        config = dict(c_prime=self.c_prime)
        base_config = super(AdaptiveFastAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":

    def test_encoder():
        print("VALIDATING ENCODER SHAPE")
        input_shape = [256, 256, 4]
        layer_filters = [32, 64, 128, 256]

        enc = encoder(input_shape, layer_filters)

        inputs = np.ones([1, 256, 256, 4], dtype=np.float32)
        enc_outs = enc(inputs)

        expected_out_shapes = [
            (1, 128, 128, 32),
            (1, 64, 64, 64),
            (1, 32, 32, 128),
            (1, 16, 16, 256),
        ]

        for o1, o2 in zip(enc_outs, expected_out_shapes):
            assert o1.shape == o2
        print("VALIDATION COMPLETE")

    def test_semantic_decoder():
        print("VALIDATING SEMANTIC DECODER SHAPE")

        input_shape = [256, 256, 4]
        layer_filters = [32, 64, 128, 256]
        enc = encoder(input_shape, layer_filters)
        inputs = np.ones([1, 256, 256, 4], dtype=np.float32)
        enc_outs = enc(inputs)

        output_shape = [256, 256]

        sem_dec = semantic_decoder(output_shape, layer_filters, 1)

        semantic_outs = sem_dec(list(reversed(enc_outs)))

        expected_out_shape = (1, 256, 256, 1)

        assert semantic_outs.shape == expected_out_shape
        print("VALIDATION COMPLETE")

    def test_instance_decoder():
        # INSTANCE DECODER =========================================================
        print("VALIDATING INSTANCE DECODER SHAPE")

        input_shape = [256, 256, 4]
        layer_filters = [32, 64, 128, 256]
        enc = encoder(input_shape, layer_filters)
        inputs = np.ones([1, 256, 256, 4], dtype=np.float32)
        enc_outs = enc(inputs)


        output_shape = [256, 256, 4]

        sem_dec = instance_decoder(output_shape, layer_filters,)

        instance_outs = sem_dec(list(reversed(enc_outs)))

        expected_out_shapes = [
            (1, 256, 256, 4, 8, 2),
            (1, 256, 256, 4, 8),
            (1, 256, 256, 1)
        ]

        for o1, o2 in zip(instance_outs, expected_out_shapes):
            assert o1.shape == o2

        print("VALIDATION COMPLETE")

    def test_instance_decoder_v2():
        # INSTANCE DECODER =========================================================
        print("VALIDATING INSTANCE DECODER V2 SHAPE")

        input_shape = [256, 256, 4]
        layer_filters = [32, 64, 128, 256]
        n = 5
        enc = encoder(input_shape, layer_filters)
        inputs = np.ones([1, 256, 256, 4], dtype=np.float32)
        enc_outs = enc(inputs)


        output_shape = [256, 256, 4]

        sem_dec = instance_decoder_v2(output_shape, layer_filters,n)

        instance_outs = sem_dec(list(reversed(enc_outs)))

        expected_out_shapes = [
            (1, 256, 256, 4, n),
            (1, 256, 256, 1)
        ]

        for o1, o2 in zip(instance_outs, expected_out_shapes):
            assert o1.shape == o2

        print("VALIDATION COMPLETE")
        # INSTANCE DECODER =========================================================
        # INSTANCE DECODER =========================================================

    def test_instance_decoder_v3():
        print("VALIDATING INSTANCE DECODER V3 SHAPE")
        input_shape = [256, 256, 4]
        layer_filters = [32, 64, 128, 256]
        enc = encoder(input_shape, layer_filters)
        inputs = np.ones([1, 256, 256, 4], dtype=np.float32)
        enc_outs = enc(inputs)


        output_shape = [256, 256, 4]

        sem_dec = instance_decoder_v3(output_shape, layer_filters)

        instance_outs = sem_dec(list(reversed(enc_outs)))

        expected_out_shapes = [
            (1, 256, 256, 4, 8),
            (1, 256, 256, 4, 8),
            (1, 256, 256, 1)
        ]

        for o1, o2 in zip(instance_outs, expected_out_shapes):
            assert o1.shape == o2
        print("VALIDATION COMPLETE")


    def test_end_to_end():
        print("VALIDATING END TO END SHAPE")

        input_shape = [256, 256, 4]
        layer_filters = [32, 64, 128, 256]

        enc = encoder(input_shape, layer_filters)

        output_shape = [256, 256]
        sem_dec = semantic_decoder(output_shape, layer_filters, 1)

        output_shape = [256, 256, 4]
        inst_dec = instance_decoder(output_shape, layer_filters,)


        in_tensor = layers.Input(shape=input_shape)
        enc_outputs = enc(in_tensor)

        reversed_outputs = list(reversed(enc_outputs))

        bkg = sem_dec(reversed_outputs)
        cv, cm, com = inst_dec(reversed_outputs)

        end_to_end = Model([in_tensor], [bkg, cv, cm, com])

        inputs = np.ones([1, 256, 256, 4], dtype=np.float32)
        outputs = end_to_end(inputs)

        expected_out_shapes = [
            (1, 256, 256, 8),
            (1, 256, 256, 8),
            (1, 256, 256, 1)
        ]

        for o1, o2 in zip(outputs, expected_out_shapes):
            assert o1.shape == o2, f"{o1.shape}, {o2}"

        print("VALIDATION COMPLETE")


    #test_encoder()
    #test_semantic_decoder()
    #test_instance_decoder()
    #test_instance_decoder_v2()
    test_instance_decoder_v3()
    #test_end_to_end()
