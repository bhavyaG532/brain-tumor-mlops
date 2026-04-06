from fastapi import FastAPI, UploadFile
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

Instrumentator().instrument(app).expose(app)

model = tf.keras.models.load_model("model.h5")

@app.post("/predict")
def predict(file: UploadFile):
    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = Image.open(path).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    return {
        "prediction": "Tumor" if pred > 0.5 else "No Tumor",
        "confidence": float(pred)
    }
