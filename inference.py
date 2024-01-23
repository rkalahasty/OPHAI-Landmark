import numpy as np
import skimage.io as io
from PIL import Image
import tensorflow as tf
import argparse
import os
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# Argument parser for inference script
parser = argparse.ArgumentParser(description='Inference script for trained models.')

# Add necessary arguments. You might need to adjust these based on your specific use case
parser.add_argument('--model_name', help='Name of the trained model.', required=True)
parser.add_argument('--model_path', help='Path to the saved model file.', required=True)
parser.add_argument('--image_path', help='Path to the image for inference.', required=True)
parser.add_argument('--output_path', help='Path to save the inference output.', default='./')
parser.add_argument('--img_size', type=int, help='Size of the image for model input.', default=512)

args = parser.parse_args()

def preprocess_image(image_path, img_size):

    img = io.imread(image_path)
    img = np.asarray(img)
    l, r = 0, img.shape[1] - 1
    while l < img.shape[1] and not any([x[0] > 100 for x in img[:, l]]):
        l += 1
    while r >= 0 and not any([x[0] > 100 for x in img[:, r]]):
        r -= 1
    t, b = 0, img.shape[0] - 1
    while t < img.shape[0] and not any([x[0] > 100 for x in img[t]]):
        t += 1
    while b >= 0 and not any([x[0] > 100 for x in img[b]]):
        b -= 1
    try:
        img = img[t:b, l:r]
        img = Image.fromarray(img).resize((img_size, img_size))
        img = np.asarray(img)
    except ValueError:
        return None
    img = img / 255.0
    print(img.shape)

    return tf.convert_to_tensor(img, dtype=tf.float32)

model = load_model(args.model_path)
input_image = preprocess_image(args.image_path, args.img_size)
input_image = np.expand_dims(input_image, axis=0)
predictions = model.predict(input_image)
np.save(os.path.join(args.output_path, 'inference_output.npy'), predictions)
print("Inference complete. Output saved to:", os.path.join(args.output_path, 'inference_output.npy'))
