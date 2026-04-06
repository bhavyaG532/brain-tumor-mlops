import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

data_dir = "data"
y_true = []
y_pred = []
confidences = []

for label in ["yes", "no"]:
    folder = os.path.join(data_dir, label)
    
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = Image.open(path).resize((224,224))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0][0]

        predicted_label = 1 if pred > 0.5 else 0
        true_label = 1 if label == "yes" else 0

        y_true.append(true_label)
        y_pred.append(predicted_label)
        confidences.append(pred)

# 📊 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
plt.title("Confusion Matrix")
plt.savefig("Results/confusion_matrix.png")
plt.clf()

# 📊 Confidence Distribution
plt.hist(confidences, bins=20)
plt.title("Prediction Confidence Distribution")
plt.savefig("Results/confidence_distribution.png")

print("Plots saved in Results/")
