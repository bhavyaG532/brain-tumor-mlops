from fastapi import FastAPI, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Metrics
Instrumentator().instrument(app).expose(app)

# Load trained model
model = tf.keras.models.load_model("model.h5")

@app.get("/")
def home():
    return {"message": "Brain Tumor Detection API is running"}

@app.post("/predict")
def predict(file: UploadFile):
    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = Image.open(path).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
    result = "Tumor"
    confidence = pred
else:
    result = "No Tumor"
    confidence = 1 - pred

return {
    "prediction": result,
    "confidence": float(round(confidence, 2))
}
