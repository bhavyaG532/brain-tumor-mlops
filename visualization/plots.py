import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
import tensorflow as tf

# ✅ Load model
model = tf.keras.models.load_model("model.h5")

# ✅ Fix input shape issue
model.build((None, 224, 224, 3))

data_dir = "data"

y_true = []
y_pred = []
confidences = []

# ✅ Loop through dataset
for label in ["yes", "no"]:
    folder = os.path.join(data_dir, label)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            # ✅ Load and preprocess image
            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224))

            img = np.array(img, dtype=np.float32) / 255.0

            # ✅ Skip bad images
            if img.shape != (224, 224, 3):
                continue

            # ✅ Add batch dimension
            img = np.expand_dims(img, axis=0)

            # ✅ Ensure correct shape
            img = np.reshape(img, (1, 224, 224, 3))

            # ✅ Predict
            pred = model.predict(img, verbose=0)[0][0]

            predicted_label = 1 if pred > 0.5 else 0
            true_label = 1 if label == "yes" else 0

            y_true.append(true_label)
            y_pred.append(predicted_label)
            confidences.append(pred)

        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue


# 📊 CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("Results/confusion_matrix.png")
plt.close()


# 📊 CONFIDENCE DISTRIBUTION
plt.figure(figsize=(6, 5))
plt.hist(confidences, bins=20)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.savefig("Results/confidence_distribution.png")
plt.close()

print("✅ Plots saved in Results/ folder")
