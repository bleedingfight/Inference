import tensorrt as trt

import numpy as np
import tensorrt as trt
from cuda import cudart
import os
logger = trt.Logger(trt.Logger.WARNING)
def onnx_to_engine(onnx_filename,input_name,min_shape,opt_shape,max_shape,engine_filename="resnet50.plan"):
    if os.path.exists(engine_filename):
        return
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    successed = parser.parse_from_file(onnx_filename)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name,min_shape,opt_shape,max_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network,config)

    with open(engine_filename, "wb") as f:
        f.write(serialized_engine)

 
if __name__ == "__main__":
    onnxFileName = "/home/liushuai9/workspace/Inference/trt-exp/resnet50-v1-12.onnx"
    min_shape = (1,3,224,224)
    opt_shape = (1,3,224,224)
    max_shape = (1,3,224,224)
    onnx_to_engine(onnxFileName,"data",min_shape,opt_shape,max_shape)

