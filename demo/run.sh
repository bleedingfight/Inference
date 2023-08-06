trtexec --onnx=resnet50-v1-12.onnx --minShapes=data:1x3x224x224 --optShapes=data:16x3x224x224 --maxShapes=data:32x3x224x224 --saveEngine=resnet50.engine
