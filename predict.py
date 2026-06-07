import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128

model = load_model("models/best_model.keras", compile=False)

image_path = "test_images/person.png"   # replace with your image

img = cv2.imread(image_path)
original = img.copy()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

mask = model.predict(img)[0]

mask = (mask > 0.5).astype(np.uint8)
mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

mask = np.expand_dims(mask, axis=-1)

result = original * mask

cv2.imwrite("output.png", result)

print("Saved output.png")

img = cv2.imread(image_path)

if img is None:
    print("Image not found:", image_path)
    exit()