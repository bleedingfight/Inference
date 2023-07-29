#!/bin/bash
if [ ! -d build ]; then
	mkdir -p build
fi
pushd build >/dev/null
cmake .. -DTENSORRT_HOME=/usr/local/TensorRT-8.6.1.6 -DCUDA_HOME=/opt/cuda -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=DEBUG
make -j16
popd >/dev/null
