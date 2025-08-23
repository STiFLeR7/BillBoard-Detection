import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load ONNX
onnx_model = onnx.load("models/best.onnx")

# Convert to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph("models/best_saved_model")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("models/best_saved_model")
tflite_model = converter.convert()

# Save
with open("models/best.tflite", "wb") as f:
    f.write(tflite_model)
