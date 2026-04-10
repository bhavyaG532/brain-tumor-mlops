from dataset import load_data
from model import build_model
import matplotlib.pyplot as plt

# 1. Load data
train, val, class_names = load_data("data/")

print("Classes:", class_names)
print("Train batches:", len(train))
print("Val batches:", len(val))

# 2. Build model
model = build_model()

# 3. Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Train
history = model.fit(
    train,
    validation_data=val,
    epochs=20
)

# 5. Print results
print(history.history)

# 6. Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train', 'val'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# 7. Save model
model.save("model.h5")
