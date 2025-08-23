from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import onnxruntime as ort

app = FastAPI()

# Load ONNX model
session = ort.InferenceSession("models/best.onnx", providers=["CPUExecutionProvider"])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess: resize to 640x640
    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized.transpose(2,0,1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Inference
    outputs = session.run(None, {"images": img_input})

    return {"detections": outputs[0].tolist()}
