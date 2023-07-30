import numpy as np
import onnx
from onnx import numpy_helper


def pb_to_array(pb_input):
    tensor = onnx.TensorProto()
    with open(pb_input, "rb") as f:
        tensor.ParseFromString(f.read())
    in_array = numpy_helper.to_array(tensor)
    return in_array


def array_to_pb(in_array, save="tensor.pb"):
    tensor = numpy_helper.from_array(in_array)
    print("TensorProto:\n{}".format(tensor))

    # Convert the TensorProto to a Numpy array
    new_array = numpy_helper.to_array(tensor)
    print("After round trip, Numpy array:\n{}\n".format(new_array))

    # Save the TensorProto
    with open(save, "wb") as f:
        f.write(tensor.SerializeToString())


in_data = pb_to_array("input_0.pb")
array_to_pb(in_data)
out_data = pb_to_array("tensor.pb")
print(in_data - out_data)
