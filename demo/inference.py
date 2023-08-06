import os
import time

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
import torchvision.transforms as transforms
from google.protobuf import json_format, text_format
from PIL import Image


def cost_time(func):
    def wrap(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        print("{} cost time:{:.5f}(s)".format(func.__name__, end - start))
        return out

    return wrap


def image_with_pil(
    image_filename, scale=1 / 255, size=(224, 224), mean=[0.485, 0.456, 0.406]
):
    image = Image.open(image_filename).convert("RGB")
    image = np.array(image.resize(size)) * scale
    image = np.transpose(image, (2, 1, 0))
    return image


def num_to_labels(filenames):
    with open(filenames, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def image_with_opencv(
    image_filename, scale=1 / 255, size=(224, 224), mean=[0.485, 0.456, 0.406]
):
    image = cv2.imread(image_filename)
    blob = cv2.dnn.blobFromImage(
        image, scalefactor=scale, size=size, mean=mean, swapRB=True, crop=False
    )
    return blob


@cost_time
def inference_with_cv(
    image_filename, onnx_model="resnet50-v1-12.onnx", label_filename="synset.txt"
):
    labels = num_to_labels(label_filename)
    net = cv2.dnn.readNetFromONNX(onnx_model)
    blob = image_with_opencv(image_filename)
    net.setInput(blob)
    start_time = time.time()
    output = net.forward()
    return np.argmax(output)


@cost_time
def inference_with_onnx(
    image_filename, onnx_model="resnet50-v1-12.onnx", label_filename="synset.txt"
):
    labels = num_to_labels(label_filename)
    session = onnxruntime.InferenceSession(
        onnx_model, providers=["CPUExecutionProvider"]
    )
    output_tensor = [node.name for node in session.get_outputs()]
    input_tensor = session.get_inputs()
    blob = image_with_opencv(image_filename)
    output = session.run(output_tensor, input_feed={input_tensor[0].name: blob})
    return np.argmax(output)


def inference_with_tensorrt(
    image_filename, onnx_model="resnet50-v1-12.onnx", label_filename="synset.txt"
):
    pass


image_filename = "kitten.jpg"
# blob = image_with_opencv(image_filename).squeeze()
# pil = image_with_pil(image_filename)
# diff = blob[0, :, :] - pil[0, :, :]
# print(diff)
out_cv = inference_with_cv(image_filename)
out_onnx = inference_with_onnx(image_filename)
print(f"opencv:{out_cv}")
print(f"onnx:{out_onnx}")
