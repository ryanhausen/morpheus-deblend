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
import os
import sys

import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras import layers
from tensorflow.keras.models import Model


import src.models.PanopticFastAttention as pfa

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")

def main(model_id:str):
    model_dir = os.path.join(MODELS_DIR, model_id)

    model_input_shape = [256, 256, 1]

    encoder_filters = [16, 32, 64, 64]
    encoder_input_shape = [256, 256, 1]
    encoder_dropout_rate = 0.1

    instance_decoder_output_shape =  [256, 256, 1]
    instance_decoder_filters = [16, 32, 64, 64]
    instance_decoder_dropout_rate = 0.1
    instance_decoder_n_instaces = 3

    inputs = layers.Input(shape=model_input_shape)

    enc = pfa.encoder(
        encoder_input_shape,
        encoder_filters,
        dropout_rate=encoder_dropout_rate
    )

    dec_intance = pfa.instance_decoder_v8(
        instance_decoder_output_shape,
        instance_decoder_filters,
        dropout_rate=instance_decoder_dropout_rate,
        n_instances=instance_decoder_n_instaces
    )

    enc_outputs = enc(inputs)
    reversed_outputs = list(reversed(enc_outputs))

    cv, cm, com = dec_intance(reversed_outputs)

    model = Model([inputs], [cv, cm, com])

    checkpoint = tf.train.Checkpoint(model=model)

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=os.path.join(model_dir, "raw"),
        max_to_keep=3,
    )

    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()

    js_dir = os.path.join(model_dir, f"js-{model_id}")
    if not os.path.exists(js_dir):
        os.mkdir(js_dir)

    h5_dir = os.path.join(model_dir, "h5")
    if not os.path.exists(h5_dir):
        os.mkdir(h5_dir)

    tf_path = os.path.join(model_dir, f"savedmodel-{model_id}")
    if not os.path.exists(tf_path):
        os.mkdir(tf_path)

    tfjs.converters.save_keras_model(model, js_dir)
    model.save(
        os.path.join(h5_dir, f"morpheus-deblend-{model_id}.h5"),
        save_format="h5",
        include_optimizer=False,
    )

    model.save(tf_path, include_optimizer=False)


if __name__=="__main__":
    model_id = sys.argv[1]
    main(model_id)





