from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np

model_loaded = keras.models.load_model("model/model.h5")

images = np.load('dataset_cleaned/eye_images.npy')

print(images.shape)

X = np.expand_dims(images, axis=3)

X_expanded= np.expand_dims(X[60000],axis=0)

print(X_expanded.shape)

print(model_loaded.predict(X_expanded)[0][0])
