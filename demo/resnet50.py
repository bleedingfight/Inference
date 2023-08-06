import torchvision.models as models
import torch
import torch.onnx

resnext50_32x4d = models.resnext50_32x4d(pretrained=True)

output = resnext50_32x4d();
BATCH_SIZE = 64
dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)
torch.onnx.export(resnext50_32x4d, dummy_input, "resnet50.onnx", verbose=False)
