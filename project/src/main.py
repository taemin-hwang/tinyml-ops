import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os

os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

model = tf.keras.models.load_model('../models/mnist_dnn_28x28.h5')
#model.summary()

img = Image.open('images.jpeg')
img_gray = img.convert('L')
img_gray = img_gray.resize((28, 28))
img_gray = np.reshape(img_gray, (1, 784))

res = model.predict(img_gray)
print(res)
