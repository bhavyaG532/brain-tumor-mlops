from dataset import load_data
from model import build_model

# 1. Load data
train, val = load_data("data/")

# ⚠️ Remove this if error comes
# print("Classes:", train.class_names)

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
    epochs=20   # 🔥 increased from 10 → better learning
)

# 5. Save model
model.save("model.h5")
