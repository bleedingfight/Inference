# Inference
TensorRT实现的推理，当前版本为C++。本机环境为：
- OS：5.8.16-2-MANJARO
- TensorRT：TensorRT-7.2.0.14
- CUDA 11.0
- NVIDIA Driver:450.80.02

# 当前实现MNIST load和推理
```
mkdir build
cd build
cmake ..
make -j2
./Inference
```

