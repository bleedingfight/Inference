import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
engine_file = 'resnet50.engine'
with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    __import__('pdb').set_trace()
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)),
                                    dtype=trt.nptype(trt.float32))
    print(h_input)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)),
                                     dtype=trt.nptype(trt.float32))

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    with engine.create_execution_context() as context:
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async(bindings=[int(d_input),
                                        int(d_output)],
                              stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        print(np.argmax(h_output))
