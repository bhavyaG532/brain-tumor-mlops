from dataset import load_data
from model import build_model
import matplotlib.pyplot as plt

# 1. Load data
train, val = load_data("data/")

# 2. Build model
model = build_model()

# 3. Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Train model
history = model.fit(
    train,
    validation_data=val,
    epochs=20
)

# 5. 🔥 Print training results
print(history.history)

# 6. 🔥 Plot accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train', 'val'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# 7. Save model
model.save("model.h5")
