# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from openvino.tools.mo.convert_impl import _convert

InputCutInfo = namedtuple("InputInfo", ["name", "shape", "type", "value"])
LayoutMap = namedtuple("LayoutMap", ["source_layout", "target_layout"])


def convert_model(input_model=None, **args):
    """
    Converts the model from original framework to OpenVino Model.

    Args:
        input_model:
            Model object in original framework (PyTorch, Tensorflow) or path to model file.
            Tensorflow*: a file with a pre-trained model (binary or text .pb file after freezing).
            Caffe*: a model proto file with model weights

            Supported formats of input model:

            PyTorch
            torch.nn.Module
            torch.jit.ScriptModule
            torch.jit.ScriptFunction

            TF
            tf.compat.v1.GraphDef
            tf.compat.v1.wrap_function
            tf.compat.v1.session

            TF2 / Keras
            tf.keras.Model
            tf.keras.layers.Layer
            tf.function
            tf.Module
            tf.train.checkpoint
            tf.python.training.tracking.base.Trackable for case when it is output from tf.saved_model.load()

    Run convert(help=true) to list all available parameters.

    Returns:
        openvino.runtime.Model
    """
    args.update({'input_model': input_model})
    return _convert(**args)
