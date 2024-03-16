from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import random
import os
from tensorflow.keras.models import load_model

model = load_model("models/model1.h5")
labels = os.listdir("data/Train")
label_map = {i: label for i, label in enumerate(labels)}

def predict_single(model, x_random, y_random, label_map):
  predictions = model.predict(x_random)
  for i in range(len(x_random)):
    print(f"Actual: {label_map[np.argmax(y_random[i])]}")
    print(f"Predicted: {label_map[np.argmax(predictions[i])]}")
    print()

with open('x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

def predict_random_indices(num):
  random_indices = random.sample(range(len(x_test)), num)
  x_random = x_test[random_indices]
  y_random = y_test[random_indices]

  print("Skin Cancer Model")
  predict_single(model, x_random, y_random, label_map)