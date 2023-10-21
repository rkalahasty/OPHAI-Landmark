import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from PIL import Image
import skimage.io as io

def process(x, xcoord, ycoord, imgsize):
    img = io.imread(x)
    # plt.imshow(img)
    # plt.show()
    img = np.asarray(img)
    l, r = 0, img.shape[1] - 1
    Image_heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    Image_heatmap[::] = 0
    Image_heatmap[ycoord, xcoord] = 1
    Image_heatmap = gaussian_filter(Image_heatmap, sigma=18)
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
        img = Image.fromarray(img).resize((imgsize, imgsize))
        img = np.asarray(img)
    except ValueError:
        return [], []

    Image_heatmap = Image_heatmap[t:b, l:r]
    Image_heatmap = Image.fromarray(Image_heatmap).resize((imgsize, imgsize))
    Image_heatmap = np.asarray(Image_heatmap)
    return img, Image_heatmap

def get_gens(img_shape, paths, test_paths, batch_size, val_size):
    paths_idx = np.random.permutation(np.arange(len(paths)))
    thres = int(len(paths)*(1-val_size))
    tp, vp = [paths[x] for x in paths_idx[:thres]], [paths[x] for x in paths_idx[thres:]]
    return fundus_gen(tp, batch_size, img_shape), fundus_gen(vp, batch_size, img_shape), fundus_gen(test_paths, batch_size, img_shape)

def fundus_gen(paths, batch_size, img_size):
    while True:
        batch_paths = [paths[i] for i in np.random.choice(a=np.arange(len(paths)), size=batch_size)]
        batch_img = []
        batch_heatmaps = []
        for tup in batch_paths:
            img_path = tup[0]
            foveaCoords = tup[1]
            discCoords = tup[2]
            img, Fovea = process(img_path, int(foveaCoords[0]), int(foveaCoords[1]), img_size)
            img, Disk = process(img_path, int(discCoords[0]), int(discCoords[1]), img_size)
            if img != [] and Fovea != [] and Disk != []:
                batch_img.append(tf.convert_to_tensor(img))
                hp = np.dstack([Fovea, Disk])
                batch_heatmaps.append(tf.convert_to_tensor(hp))
                # plt.imshow(Fovea)
                # plt.show()
                # plt.imshow(img)
                # plt.show()
        batch_img = np.array(batch_img)
        batch_heatmaps = np.array(batch_heatmaps)
        yield (tf.convert_to_tensor(batch_img), tf.convert_to_tensor(batch_heatmaps))
