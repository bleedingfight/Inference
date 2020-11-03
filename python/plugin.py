import sys
import os
import ctypes
from random import randint

from PIL import Image
import numpy as np
import tensorflow as tf

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
import graphsurgeon as gs
import uff

# ../common.py
sys.path.insert(1,
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    os.pardir
                )
                )
sys.path.insert(1,
                os.path.join(
                    "/usr/local/TensorRT-7.2.0.14/samples",
                    "python"
                )
                )
import common

# lenet5.py
# import lenet5


# MNIST dataset metadata
MNIST_IMAGE_SIZE = 28
MNIST_CHANNELS = 1
MNIST_CLASSES = 10

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# Define some global constants about the model.
class ModelData(object):
    INPUT_NAME = "InputLayer"
    INPUT_SHAPE = (MNIST_CHANNELS, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
    LRELU_NAME = "LReLU"
    OUTPUT_NAME = "OutputLayer/Softmax"
    OUTPUT_SHAPE = (MNIST_IMAGE_SIZE, )
    DATA_TYPE = trt.float32


# Generates mappings from unsupported TensorFlow operations to TensorRT plugins
def prepare_namespace_plugin_map():
    # In this sample, the only operation that is not supported by TensorRT
    # is tf.nn.leak_relu, so we create a new node which will tell UffParser which
    # plugin to run and with which arguments in place of tf.nn.leak_relu.


    # The "clipMin" and "clipMax" fields of this TensorFlow node will be parsed by createPlugin,
    # and used to create a CustomClipPlugin with the appropriate parameters.
    trt_lrelu = gs.create_plugin_node(name="trt_lrelu", op="CustomClipPlugin", negSlope=0.7)
    namespace_plugin_map = {
        ModelData.LRELU_NAME: trt_lrelu
    }
    return namespace_plugin_map


# Builds TensorRT Engine
def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = common.GiB(1)

        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_path, network)

        return builder.build_cuda_engine(network)
def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.reshape(x_train, (-1, 1, 28, 28))
    x_test = np.reshape(x_test, (-1, 1, 28, 28))
    return x_train, y_train, x_test, y_test

# Loads a test case into the provided pagelocked_buffer. Returns loaded test case label.
def load_normalized_test_case(pagelocked_buffer):
    _, _, x_test, y_test = load_data()
    num_test = len(x_test)
    case_num = randint(0, num_test-1)
    img = x_test[case_num].ravel()
    np.copyto(pagelocked_buffer, img)
    return y_test[case_num]

def main():
    CLIP_PLUGIN_LIBRARY = os.path.join("/home/liushuai/Inference/cmake-build-debug/plugin","libclipplugin.so")
    ctypes.CDLL(CLIP_PLUGIN_LIBRARY)
    MODEL_PATH = os.path.join("/home/liushuai/Inference/python/models","trained_lenet5.uff")

    # Build an engine and retrieve the image mean from the model.
    with build_engine(MODEL_PATH) as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            print("\n=== Testing ===")
            test_case = load_normalized_test_case(inputs[0].host)
            print("Loading Test Case: " + str(test_case))
            # The common do_inference function will return a list of outputs - we only have one in this case.
            [pred] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print("Prediction: " + str(np.argmax(pred)))


if __name__ == "__main__":
    main()