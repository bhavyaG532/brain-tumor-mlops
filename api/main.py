from fastapi import FastAPI, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# ✅ Metrics (Prometheus)
Instrumentator().instrument(app).expose(app)

# ✅ Load trained model
model = tf.keras.models.load_model("model.h5")


# ✅ Home route
@app.get("/")
def home():
    return {"message": "Brain Tumor Detection API is running"}


# ✅ Prediction route
@app.post("/predict")
def predict(file: UploadFile):

    # Save uploaded file temporarily
    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and preprocess image
    img = Image.open(path).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Model prediction
    pred = model.predict(img)[0][0]

    print("Raw prediction:", pred)  # ✅ debug

    # Convert to label
    if pred > 0.5:
        result = "Tumor"
        confidence = float(pred)
    else:
        result = "No Tumor"
        confidence = float(1 - pred)

    # Remove temp file
    os.remove(path)

    return {
        "prediction": result,
        "confidence": confidence   # ❌ no rounding
    }
