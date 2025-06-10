import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist

# 載入與正規化資料
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 建立模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 編譯與訓練
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_split=0.1)

# 儲存為 .h5
model.save("fashion_model.h5")

from tensorflow.keras.models import load_model
import numpy as np
import json

model = load_model("fashion_mnist.h5")

# 儲存 architecture
with open("fashion_mnist.json", "w") as json_file:
    json_file.write(model.to_json())

# 儲存權重
weights = {}
for i, layer in enumerate(model.layers):
    for j, weight in enumerate(layer.get_weights()):
        weights[f"layer_{i}_{j}"] = weight

np.savez("fashion_mnist.npz", **weights)
