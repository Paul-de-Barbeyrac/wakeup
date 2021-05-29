import numpy as np
from skimage.transform import resize
import cv2
import glob


def generate_clean_dataset():
    images = []
    y = []

    for filepath in glob.iglob('dataset/*/*.png'):
        img = cv2.imread(filepath)
        image_resized = resize(img, (28, 28), anti_aliasing=True)
        if img is not None:  # in case unable to open file
            images.append(image_resized)
            y.append(int(filepath.split("_")[4]))  # Retrieve eye state from file naming convention

    output_images = [image[:, :, 0] for image in images]

    # Dump file for faster reuse
    np.save('dataset_cleaned/eye_images.npy', output_images)
    np.save('dataset_cleaned/eye_states.npy', y)


if __name__ == '__main__':
    generate_clean_dataset()
