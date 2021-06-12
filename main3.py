import matplotlib.image as mpimg
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow import keras

model_loaded = keras.models.load_model("model/model.h5")

mark_img = mpimg.imread('right-eye-192-252.jpg')

mark_img = rgb2gray(mark_img)
mark_img = resize(mark_img, (28, 28))
mark_img = np.expand_dims(mark_img, axis=2)
mark_img = np.expand_dims(mark_img, axis=0)
print(mark_img.shape)

y_pred = model_loaded.predict(mark_img)
print(y_pred)
