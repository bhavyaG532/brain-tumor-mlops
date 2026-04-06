import os
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

def predict_folder(folder):
    results = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = Image.open(path).resize((224,224))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0][0]

        results.append((file, pred))

    return results
