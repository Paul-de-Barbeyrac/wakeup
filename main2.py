from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np

model_loaded = keras.models.load_model("model/model.h5")

images = np.load('dataset_cleaned/eye_images.npy')

print(images.shape)
print(model_loaded.predict(images[10]))

