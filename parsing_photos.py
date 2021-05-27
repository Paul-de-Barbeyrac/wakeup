import cv2
import glob

images = []
for filepath in glob.iglob('dataset/*/*.png'):
    img = cv2.imread(filepath)
    if img is not None:
        images.append(img)

print(len(images))

print(images[0])
