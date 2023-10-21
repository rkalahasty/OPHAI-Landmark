from tqdm import tqdm
import os
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
from tensorflow.python.ops.numpy_ops import np_config
import dlib
import cv2
import imutils
import multiprocessing

#
np_config.enable_numpy_behavior()
tf.data.experimental.enable_debug_mode()
tf.compat.v1.enable_eager_execution()

import sys

sys.path.insert(0, '/OPHAI-Landmark-Localization-main/ophai')
sys.path.append('../ophai/')

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

import pandas as pd
import io

print(tf.__version__)

os.chdir("C:\\Users\\17033\\Desktop\\data_0")
df_train = pd.read_csv("data_RIGA_train_LandmarkAddition.csv")
df_test = pd.read_csv("data_RIGA_test_LandmarkAddition.csv")

import tifffile as tif
from PIL import Image

X_train, y_train, X_test, y_test = [], [], [], []

total = df_train.shape[0]
counter = 0


def process(x, xcoord, ycoord):
    img = tif.imread(x) if '.tif' in x else cv2.imread(x)
    img = np.asarray(img)
    l, r = 0, img.shape[1] - 1
    Image_heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    Image_heatmap[::] = 0
    Image_heatmap[ycoord, xcoord] = 1
    Image_heatmap = gaussian_filter(Image_heatmap, sigma=20)
    Image_heatmap = (Image_heatmap / np.max(Image_heatmap))

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
        img = Image.fromarray(img).resize((256, 256))
        img = np.asarray(img)
    except ValueError:
        return "Error", 0

    Image_heatmap = Image_heatmap[t:b, l:r]
    Image_heatmap = Image.fromarray(Image_heatmap).resize((256, 256))
    Image_heatmap = np.asarray(Image_heatmap)
    return img, Image_heatmap


if not os.path.exists("dlib_landmarks"): os.mkdir("dlib_landmarks")

os.chdir("dlib_landmarks")
output = open("landmark_localization.xml", "w")
output.write("<?xml version='1.0' encoding='ISO-8859-1'?> \n")
output.write("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?> \n")
output.write("<dataset> \n")
output.write("<name>Training Eyes</name> \n")
output.write("<images> \n")

for r in tqdm(df_train.iterrows()):
    imagePath = r[1]["imageDIR"] + "\\" + r[1]["imageID"]
    if r[1]["Fovea_X"] != "N":
        os.chdir("C:\\Users\\17033\\Desktop\\data_0")
        img1, Fovea = process(imagePath, int(r[1]["Fovea_X"]), int(r[1]["Fovea_Y"]))
        img2, Cup = process(imagePath, int(r[1]["Cup_X"]), int(r[1]["Cup_Y"]))
        img3, Disk = process(imagePath, int(r[1]["Disc_X"]), int(r[1]["Disc_Y"]))
        os.chdir("dlib_landmarks")
        if img1 != "Error" and img2 != "Error" and img3 != "Error":
            new_image_path = os.path.basename(imagePath[:imagePath.index(".")]) + ".png"
            im = Image.fromarray(img1)
            im.save(new_image_path)
            output.write(f"<image file='{new_image_path}'> \n")
            output.write(f"<box top='1' left='1' width='255' height='255'> \n")
            coordsf = np.unravel_index(Fovea.argmax(), Fovea.shape)[:2]
            output.write(f"<part name='Fovea' x='{coordsf[0]}' y='{coordsf[1]}'/> \n")
            coordsC = np.unravel_index(Cup.argmax(), Cup.shape)[:2]
            output.write(f"<part name='Cup' x='{coordsC[0]}' y='{coordsC[1]}'/> \n")
            coordsD = np.unravel_index(Disk.argmax(), Disk.shape)[:2]
            output.write(f"<part name='Disk' x='{coordsD[0]}' y='{coordsD[1]}'/> \n")
            counter += 1
            output.write(f"</box> \n")
            output.write(f"</image> \n")

output.write("</images> \n")
output.write("</dataset> \n")
