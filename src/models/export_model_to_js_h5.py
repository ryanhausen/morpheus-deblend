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

    model_input_shape = [256, 256, 4]

    encoder_filters = [32, 64, 128, 256]
    encoder_input_shape = [256, 256, 4]

    instance_decoder_output_shape =  [256, 256, 4]
    instance_decoder_filters = [32, 64, 128, 256]

    semantic_decoder_output_shape = [256, 256, 4]
    semantic_decoder_filters = 	[32, 64, 128, 256]
    semantic_decoder_n_classes = 1

    inputs = layers.Input(shape=model_input_shape)

    enc = pfa.encoder(encoder_input_shape, encoder_filters)
    dec_semantic = pfa.semantic_decoder(
        semantic_decoder_output_shape,
        semantic_decoder_filters,
        semantic_decoder_n_classes
    )
    dec_intance = pfa.instance_decoder(
        instance_decoder_output_shape,
        instance_decoder_filters
    )

    enc_outputs = enc(inputs)
    reversed_outputs = list(reversed(enc_outputs))

    bkg = dec_semantic(reversed_outputs)
    cv, cm, com = dec_intance(reversed_outputs)

    model = Model([inputs], [bkg, cv, cm, com])

    checkpoint = tf.train.Checkpoint(model=model)

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=os.path.join(model_dir, "raw"),
        max_to_keep=3,
    )

    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()

    js_dir = os.path.join(model_dir, "js")
    if not os.path.exists(js_dir):
        os.mkdir(js_dir)

    h5_dir = os.path.join(model_dir, "h5")
    if not os.path.exists(h5_dir):
        os.mkdir(h5_dir)

    tfjs.converters.save_keras_model(model, js_dir)
    model.save(
        os.path.join(h5_dir, "morpheus-deblend.h5"),
        save_format="h5",
        include_optimizer=False,
    )


if __name__=="__main__":
    model_id = sys.argv[1]
    main(model_id)





