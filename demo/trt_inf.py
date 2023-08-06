import os
import subprocess

import numpy as np
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
model_path = "resnet50-v1-12.onnx"


def preprocess_onnx(onnx_path):
    if not os.path.exists(model_path):
        p = subprocess.Popen(
            "wget -c https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12.onnx",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = p.communicate()


network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

parser = trt.OnnxParser(network, logger)
success = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

if not success:
    logger.error("=================")
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 25)  # 1 MiB
profiler = builder.create_optimization_profile()
profiler.set_shape("data", (1, 3, 224, 224), (8, 3, 224, 224), (32, 3, 224, 224))
config.add_optimization_profile(profiler)
serialized_engine = builder.build_serialized_network(network, config)

plan_path = model_path.replace("onnx", "plan")
if not os.path.exists(plan_path):
    with open(plan_path, "wb") as f:
        f.write(serialized_engine)

output = network.get_output(0).name
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)

context = engine.create_execution_context()


def run_with_tensorrt(context, bUsePinnedMemory):
    io_nums = context.num_io_tensors
    assert io_nums == 2, "resnet50的输入输出个数为2，现在为:{}".format(io_nums)
    in_node = context.get_tensor_name(0)
    out_node = context.get_tensor_name(1)
    context.set_input_shape(in_node, [1, 3, 224, 224])


# context.set_tensor_address(name, ptr)
# context.execute_async_v3(buffers, stream_ptr)
