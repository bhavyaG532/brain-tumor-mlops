from dataset import load_data
from model import build_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load data
train, val, class_names = load_data("data/")

print("Classes:", class_names)
print("Train batches:", len(train))
print("Val batches:", len(val))

# 2. Build model
model = build_model()

# 3. Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Early stopping (🔥 important)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# 5. Train model
history = model.fit(
    train,
    validation_data=val,
    epochs=20,
    callbacks=[early_stop]
)

# 6. Print training results
print(history.history)

# 7. Plot accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train', 'val'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# 8. Save model
model.save("model.h5")
