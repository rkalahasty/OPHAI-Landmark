import argparse
import os
from models.hbaunet import hbaunet

parser = argparse.ArgumentParser(description='Inference script for trained models.')

parser.add_argument('--model_name', help='Name of the trained model.', required=True)
parser.add_argument('--model_path', help='Path to the saved model file.', required=True)
parser.add_argument('--image_path', help='Path to the image for inference.', required=True)
parser.add_argument('--output_path', help='Path to save the inference output.', default='./')
parser.add_argument('--img_size', type=int, help='Size of the image for model input.', default=256)

args = parser.parse_args()

import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.models import load_model
from datautils import process
import numpy as np

imgpath = args.image_path
if "hba" in args.model_name:
    model = hbaunet((args.img_size, args.img_size, 3), dropout_rate=0.4, use_attnDecoder=False, skip=False, num_layers=3)
    model.load_weights(args.model_path)
else:
    model = load_model(args.model_path)
img, Fovea = process(imgpath, 2, 2, 256)
img, Disk = process(imgpath, 2, 2, 256)

fig,a =  plt.subplots(2)
fig.tight_layout(pad=.20)
fig.suptitle('Perturbation', fontsize=16)
fig.subplots_adjust(top=0.88)
for i in range(1):
    plt.setp(a[i].get_xticklabels(), visible=False)
    plt.setp(a[i].get_yticklabels(), visible=False)
    a[i].set_xticks([])
    a[i].set_yticks([])
    a[i].tick_params(axis='both', which='both', length=0)

a[0].imshow(img)
a[0].set_title('Orig. Image', fontsize = 10)


output = model(np.array([img]))
output = np.array(output[0])
needed_multi_channel_img = np.zeros((256, 256, 3))
needed_multi_channel_img[:, :, 0] = output[:, :, 0]
needed_multi_channel_img[:, :, 1] = output[:, :, 1]
a[1].imshow(needed_multi_channel_img)
a[1].set_title('Predictions.', fontsize=10)
a[1].set_xlabel(args.model_name, fontsize=10)


plt.show()
output_file_path = os.path.join(args.output_path, 'inference_output.npy')
np.save(output_file_path, output)
print("Inference complete. Output saved to:", output_file_path)

