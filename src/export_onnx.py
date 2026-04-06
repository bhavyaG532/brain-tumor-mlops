import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

output_path = "model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

print("ONNX model saved!")
