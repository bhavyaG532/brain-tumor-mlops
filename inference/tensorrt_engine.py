# Simplified placeholder (real TRT needs GPU setup)

import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")

def predict_trt(image):
    image = image.astype(np.float32)
    inputs = {"input": image}
    outputs = session.run(None, inputs)
    return outputs
