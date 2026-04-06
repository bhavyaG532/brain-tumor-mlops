from dataset import load_data
from model import build_model

train, val = load_data("data/")

model = build_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train, validation_data=val, epochs=10)

model.save("model.h5")
