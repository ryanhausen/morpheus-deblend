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


def get_model() -> Model:
    pass


@gin.configurable()
def encoder(
    input_shape:List[int],
    layer_filters:List[int],
    name:str="MorpheusDeblendEncoder"
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
        lambda l, n: res_down(l, name_prefix=n),
        zip(layer_filters, name_prefixes)
    )

    model_outputs = []
    def apply_res(x:tf.Tensor, func:LayerFunc) -> tf.Tensor:
        f_x = func(x)
        model_outputs.append(f_x)
        return f_x


    model_input = layers.Input(shape=input_shape, name=f"{name}_input")
    reduce(apply_res, res_funcs, model_input)

    return Model([model_input], model_outputs, name=name)


@gin.configurable()
def semantic_decoder(
    output_shape:List[int],
    filters:List[int],
    n_classes:int,
    name:str="MorpheusDeblendSemanticDecoder",
) -> Model:
    """The semantic decoder module outputs a class map for semantic segmentation.

    """

    hw = output_shape[0]
    input_shapes = list(reversed([[hw//(2**i), hw//(2**i), c] for i,c in enumerate(filters, start=1)]))

    inputs = [layers.Input(shape=s) for s in input_shapes]
    att_outs = list(map(lambda x: FastAttention()(x), inputs))

    # We map the filters in reverse order because we start small and grow the
    # output back to the input resolution. We add an additional filter for the
    # final upsample and out
    fuse_funcs = list(map(lambda f: fuse_up(f, name_prefix=name), list(reversed(filters[:-1]))+[filters[-1]]))
    fuse_funcs_y = list(zip(fuse_funcs, att_outs))

    def apply_fuse(x:tf.Tensor, func_y:Tuple[LayerFunc, tf.Tensor]) -> tf.Tensor:
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
    output_shape:List[int],
    filters:List[int],
    name:str="MorpheusDeblendInstanceDecoder",
) -> Model:
    """The instance decoder outputs the values needed for source separation.

    """
    hw = output_shape[0]
    input_shapes = list(reversed([[hw//(2**i), hw//(2**i), c] for i,c in enumerate(filters, start=1)]))

    inputs = [layers.Input(shape=s) for s in input_shapes]
    att_outs = list(map(lambda x: FastAttention()(x), inputs))

    # We map the filters in reverse order because we start small and grow the
    # output back to the input resolution. We add an additional filter for the
    # final upsample and out
    fuse_funcs = list(map(lambda f: fuse_up(f, name_prefix=name), list(reversed(filters[:-1]))+[filters[-1]]))
    fuse_funcs_y = list(zip(fuse_funcs, att_outs))


    def apply_fuse(x:tf.Tensor, func_y:Tuple[LayerFunc, tf.Tensor]) -> tf.Tensor:
        func, y = func_y
        f_x = func(x, y)
        return f_x

    fuse_out = reduce(apply_fuse, fuse_funcs_y, None)
    up_out = layers.UpSampling2D()(fuse_out)

    pre_conv = partial(layers.Conv2D, 32, 5, padding="SAME")


    claim_vectors_conv = layers.Conv2D(16, 1, padding="SAME")(pre_conv()(up_out))
    claim_vectors  = layers.Reshape(output_shape + [8, 2])(claim_vectors_conv)

    claim_map = layers.Conv2D(8, 1, padding="SAME")(pre_conv()(up_out))

    center_of_mass = layers.Conv2D(1, 1, padding="SAME")(pre_conv()(up_out))

    return Model(inputs, [claim_vectors, claim_map, center_of_mass], name=name)





@gin.configurable(whitelist=["kernel_size", "activation"])
def res_down(
    filters:int,
    kernel_size:int=3,
    activation:layers.Layer=layers.ReLU,
    name_prefix:str="",
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

@gin.configurable(whitelist=["activation"])
def fuse_up(
    filters: int,
    activation:layers.Layer=layers.ReLU,
    name_prefix:str = "",
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

@gin.configurable(whitelist=["c_prime"])
class FastAttention(tf.keras.layers.Layer):
    """ Fast Attention Module from:

    Real-time Semantic Segmentation with Fast Attention
    https://arxiv.org/pdf/2007.03815.pdf


    Args:
        c_prime (int): The number of attention features in q and k
    """

    def __init__(self, c_prime:int=32, **kwargs):
        super(FastAttention, self).__init__(**kwargs)
        self.c_prime = c_prime

        self.q_conv = tf.keras.layers.Conv2D(c_prime, 1)
        self.q_bn = tf.keras.layers.BatchNormalization()
        self.k_conv = tf.keras.layers.Conv2D(c_prime, 1)
        self.k_bn = tf.keras.layers.BatchNormalization()

        self.v_bn = tf.keras.layers.BatchNormalization()

        self.att_bn = tf.keras.layers.BatchNormalization()

        self.kT_v_mul = tf.keras.layers.Dot(axes=(1,1))
        self.q_kTv_mul = tf.keras.layers.Dot(axes=(2,1))

        self.qkv_bn = tf.keras.layers.BatchNormalization()

        self.residual_add = tf.keras.layers.Add(name="FastAttentionOut")

    def build(self, input_shape): # [None, h, w, c]
        h = w = input_shape[1]
        n = input_shape[1] * input_shape[2]
        c = input_shape[-1]

        self.n = tf.constant(n, dtype=tf.float32)
        self.att_conv = tf.keras.layers.Conv2D(c, 1) # [None, h, w, c]
        self.v_conv = tf.keras.layers.Conv2D(c, 1) # [None, h, w, c]

        self.flat_q = tf.keras.layers.Reshape([n, self.c_prime])
        self.flat_k = tf.keras.layers.Reshape([n, self.c_prime])
        self.flat_v = tf.keras.layers.Reshape([n, c])

        self.square_qkv = tf.keras.layers.Reshape([h, w, c])

        self.qkv_conv = tf.keras.layers.Conv2D(c, 1)

    def call(self, inputs):

        pre_v = self.v_bn(self.v_conv(inputs)) # [None, h, w, c]

        q = K.l2_normalize(self.flat_q(self.q_bn(self.q_conv(inputs))), axis=-1) # [None, n, c']
        k = K.l2_normalize(self.flat_k(self.k_bn(self.k_conv(inputs))), axis=-1) # [None, n, c']
        v = K.relu(self.flat_v(pre_v)) #[None, n, c]

        kTv = self.kT_v_mul([k, v]) # [None, c', c]
        qkTv = self.q_kTv_mul([q, kTv]) # [None, n, c]

        square_qkTv = self.square_qkv(qkTv) # [None, h, w, c]
        qkTv_out = self.qkv_bn(self.qkv_conv(square_qkTv)) #[None, h, w, c]

        att_out = self.residual_add([pre_v, qkTv_out]) # [None, h, w, c]

        return att_out

    # Adapted from:
    # https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/layers/dense_attention.py#L483
    def get_config(self):
        config = dict(c_prime=self.c_prime)
        base_config = super(tf.keras.layers.Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




if __name__=="__main__":

    # ENCODER ==================================================================
    print("VALIDATING ENCODER SHAPE")
    input_shape = [256, 256, 4]
    layer_filters = [32, 64, 128, 256]

    enc = encoder(input_shape, layer_filters)

    inputs = np.ones([1, 256, 256, 4])
    enc_outs = enc(inputs)

    expected_out_shapes = [
        (1, 128, 128, 32),
        (1, 64, 64, 64),
        (1, 32, 32, 128),
        (1, 16, 16, 256)
    ]

    for o1, o2 in zip(enc_outs, expected_out_shapes):
        assert o1.shape==o2
    print("VALIDATION COMPLETE")
    # ENCODER ==================================================================

    # SEMANTIC DECODER =========================================================
    print("VALIDATING SEMANTIC DECODER SHAPE")
    output_shape = [256, 256]

    sem_dec = semantic_decoder(
        output_shape,
        layer_filters,
        1
    )

    semantic_outs = sem_dec(list(reversed(enc_outs)))

    expected_out_shape = (1, 256, 256, 1)

    assert semantic_outs.shape == expected_out_shape
    print("VALIDATION COMPLETE")
    # SEMANTIC DECODER =========================================================


    # INSTANCE DECODER =========================================================
    print("VALIDATING INSTANCE DECODER SHAPE")
    output_shape = [256, 256]

    sem_dec = instance_decoder(
        output_shape,
        layer_filters,
    )

    instance_outs = sem_dec(list(reversed(enc_outs)))

    expected_out_shapes = [
        (1, 256, 256, 8, 2),
        (1, 256, 256, 8),
        (1, 256, 256, 1)
    ]

    for o1, o2 in zip(instance_outs, expected_out_shapes):
        assert o1.shape==o2

    print("VALIDATION COMPLETE")
    # INSTANCE DECODER =========================================================







