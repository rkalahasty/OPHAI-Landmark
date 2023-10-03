import tensorflow as tf
print('Tensorflow version : {}'.format(tf.__version__))
print('GPU : {}'.format(tf.config.list_physical_devices('GPU')))
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Concatenate, concatenate, Dropout, LeakyReLU, Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import io
from PIL import Image
from scipy.ndimage import gaussian_filter

class SpaceToDepth(keras.layers.Layer):

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        super(SpaceToDepth, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        batch, height, width, depth = K.int_shape(x)
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = K.reshape(x, (batch, reduced_height, self.block_size,
                             reduced_width, self.block_size, depth))
        z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
        t = K.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size **2))
        return t

    def compute_output_shape(self, input_shape):
        shape =  (input_shape[0], input_shape[1] // self.block_size, input_shape[2] // self.block_size,
                  input_shape[3] * self.block_size **2)
        return tf.TensorShape(shape)

def YoloV2(input_shape, CLASS=2, BOX=5, GRID_H = 8,  GRID_W  = 8):

    input_image = tf.keras.layers.Input((input_shape[0], input_shape[1], 3), dtype='float32')

    # Layer 1
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
        skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)

    skip_connection = SpaceToDepth(block_size=2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)  # add dropout

    # Layer 23
    x = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_W, GRID_H, BOX, 4 + 1 + CLASS))(x)

    model = keras.models.Model(input_image, output)
    return model

def setWeights(weightFile, model, GRID_H = 8, GRID_W = 8):
    class WeightReader:
        def __init__(self, weight_file):
            self.offset = 4
            self.all_weights = np.fromfile(weight_file, dtype='float32')

        def read_bytes(self, size):
            self.offset = self.offset + size
            return self.all_weights[self.offset - size:self.offset]

        def reset(self):
            self.offset = 4

    weight_reader = WeightReader('yolo.weights')

    weight_reader.reset()
    nb_conv = 23

    for i in range(1, nb_conv + 1):
        conv_layer = model.get_layer('conv_' + str(i))
        conv_layer.trainable = True

        if i < nb_conv:
            norm_layer = model.get_layer('norm_' + str(i))
            norm_layer.trainable = True

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])

    layer   = model.layers[-2] # last convolutional layer
    layer.trainable = True


    weights = layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
    new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

    layer.set_weights([new_kernel, new_bias])


def iou(x1, y1, w1, h1, x2, y2, w2, h2):
    '''
    Calculate IOU between box1 and box2

    Parameters
    ----------
    - x, y : box center coords
    - w : box width
    - h : box height

    Returns
    -------
    - IOU
    '''
    xmin1 = x1 - 0.5 * w1
    xmax1 = x1 + 0.5 * w1
    ymin1 = y1 - 0.5 * h1
    ymax1 = y1 + 0.5 * h1
    xmin2 = x2 - 0.5 * w2
    xmax2 = x2 + 0.5 * w2
    ymin2 = y2 - 0.5 * h2
    ymax2 = y2 + 0.5 * h2
    interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    inter = interx * intery
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / (union + 1e-6)
    return iou


def yolov2_loss(detector_mask, matching_true_boxes, class_one_hot, true_boxes_grid, y_pred, info=False, GRID_H = 8, GRID_W = 8):
    '''
    Calculate YOLO V2 loss from prediction (y_pred) and ground truth tensors (detector_mask,
    matching_true_boxes, class_one_hot, true_boxes_grid,)

    Parameters
    ----------
    - detector_mask : tensor, shape (batch, size, GRID_W, GRID_H, anchors_count, 1)
        1 if bounding box detected by grid cell, else 0
    - matching_true_boxes : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 5)
        Contains adjusted coords of bounding box in YOLO format
    - class_one_hot : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, class_count)
        One hot representation of bounding box label
    - true_boxes_grid : annotations : tensor (shape : batch_size, max annot, 5)
        true_boxes_grid format : x, y, w, h, c (coords unit : grid cell)
    - y_pred : prediction from model. tensor (shape : batch_size, GRID_W, GRID_H, anchors count, (5 + labels count)
    - info : boolean. True to get some infox about loss value

    Returns
    -------
    - loss : scalar
    - sub_loss : sub loss list : coords loss, class loss and conf loss : scalar
    '''
    LAMBDA_NOOBJECT = 1
    LAMBDA_OBJECT = 5
    LAMBDA_CLASS = 1
    LAMBDA_COORD = 1

    # anchors tensor
    ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
    anchors = np.array(ANCHORS)
    anchors = anchors.reshape(len(anchors) // 2, 2)

    # grid coords tensor
    coord_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), tf.float32)
    coord_y = tf.transpose(coord_x, (0, 2, 1, 3, 4))
    coords = tf.tile(tf.concat([coord_x, coord_y], -1), [y_pred.shape[0], 1, 1, 5, 1])

    # coordinate loss
    pred_xy = K.sigmoid(y_pred[:, :, :, :, 0:2])  # adjust coords between 0 and 1
    pred_xy = (pred_xy + coords)  # add cell coord for comparaison with ground truth. New coords in grid cell unit
    pred_wh = K.exp(y_pred[:, :, :, :,
                    2:4]) * anchors  # adjust width and height for comparaison with ground truth. New coords in grid cell unit
    # pred_wh = (pred_wh * anchors) # unit : grid cell
    nb_detector_mask = K.sum(tf.cast(detector_mask > 0.0, tf.float32))
    xy_loss = LAMBDA_COORD * K.sum(detector_mask * K.square(matching_true_boxes[..., :2] - pred_xy)) / (
                nb_detector_mask + 1e-6)  # Non /2
    wh_loss = LAMBDA_COORD * K.sum(detector_mask * K.square(K.sqrt(matching_true_boxes[..., 2:4]) -
                                                            K.sqrt(pred_wh))) / (nb_detector_mask + 1e-6)
    coord_loss = xy_loss + wh_loss

    # class loss
    pred_box_class = y_pred[..., 5:]
    true_box_class = tf.argmax(class_one_hot, -1)
    # class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    class_loss = K.sparse_categorical_crossentropy(target=true_box_class, output=pred_box_class, from_logits=True)
    class_loss = K.expand_dims(class_loss, -1) * detector_mask
    class_loss = LAMBDA_CLASS * K.sum(class_loss) / (nb_detector_mask + 1e-6)

    # confidence loss
    pred_conf = K.sigmoid(y_pred[..., 4:5])
    # for each detector : iou between prediction and ground truth
    x1 = matching_true_boxes[..., 0]
    y1 = matching_true_boxes[..., 1]
    w1 = matching_true_boxes[..., 2]
    h1 = matching_true_boxes[..., 3]
    x2 = pred_xy[..., 0]
    y2 = pred_xy[..., 1]
    w2 = pred_wh[..., 0]
    h2 = pred_wh[..., 1]
    ious = iou(x1, y1, w1, h1, x2, y2, w2, h2)
    ious = K.expand_dims(ious, -1)

    # for each detector : best ious between prediction and true_boxes (every bounding box of image)
    pred_xy = K.expand_dims(pred_xy, 4)  # shape : m, GRID_W, GRID_H, BOX, 1, 2
    pred_wh = K.expand_dims(pred_wh, 4)
    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half
    true_boxe_shape = K.int_shape(true_boxes_grid)
    true_boxes_grid = K.reshape(true_boxes_grid, [true_boxe_shape[0], 1, 1, 1, true_boxe_shape[1], true_boxe_shape[2]])
    true_xy = true_boxes_grid[..., 0:2]
    true_wh = true_boxes_grid[..., 2:4]
    true_wh_half = true_wh * 0.5
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half
    intersect_mins = K.maximum(pred_mins, true_mins)  # shape : m, GRID_W, GRID_H, BOX, max_annot, 2
    intersect_maxes = K.minimum(pred_maxes, true_maxes)  # shape : m, GRID_W, GRID_H, BOX, max_annot, 2
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # shape : m, GRID_W, GRID_H, BOX, 1, 1
    true_areas = true_wh[..., 0] * true_wh[..., 1]  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = K.expand_dims(best_ious)  # shape : m, GRID_W, GRID_H, BOX, 1

    # no object confidence loss
    no_object_detection = K.cast(best_ious < 0.6, K.dtype(best_ious))
    noobj_mask = no_object_detection * (1 - detector_mask)
    nb_noobj_mask = K.sum(tf.cast(noobj_mask > 0.0, tf.float32))

    noobject_loss = LAMBDA_NOOBJECT * K.sum(noobj_mask * K.square(-pred_conf)) / (nb_noobj_mask + 1e-6)
    # object confidence loss
    object_loss = LAMBDA_OBJECT * K.sum(detector_mask * K.square(ious - pred_conf)) / (nb_detector_mask + 1e-6)
    # total confidence loss
    conf_loss = noobject_loss + object_loss

    # total loss
    loss = conf_loss + class_loss + coord_loss
    sub_loss = [conf_loss, class_loss, coord_loss]

    if info:
        print('conf_loss   : {:.4f}'.format(conf_loss))
        print('class_loss  : {:.4f}'.format(class_loss))
        print('coord_loss  : {:.4f}'.format(coord_loss))
        print('    xy_loss : {:.4f}'.format(xy_loss))
        print('    wh_loss : {:.4f}'.format(wh_loss))
        print('--------------------')
        print('total loss  : {:.4f}'.format(loss))

        # display masks for each anchors
        for i in range(len(anchors)):
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
            f.tight_layout()
            f.suptitle('MASKS FOR ANCHOR {} :'.format(anchors[i, ...]))

            ax1.matshow((K.sum(detector_mask[0, :, :, i], axis=2)), cmap='Greys', vmin=0, vmax=1)
            ax1.set_title('detector_mask, count : {}'.format(K.sum(tf.cast(detector_mask[0, :, :, i] > 0., tf.int32))))
            ax1.xaxis.set_ticks_position('bottom')

            ax2.matshow((K.sum(no_object_detection[0, :, :, i], axis=2)), cmap='Greys', vmin=0, vmax=1)
            ax2.set_title('no_object_detection mask')
            ax2.xaxis.set_ticks_position('bottom')

            ax3.matshow((K.sum(noobj_mask[0, :, :, i], axis=2)), cmap='Greys', vmin=0, vmax=1)
            ax3.set_title('noobj_mask')
            ax3.xaxis.set_ticks_position('bottom')

    return loss, sub_loss

def grad(model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes, training=True):
    with tf.GradientTape() as tape:
        y_pred = model(img, training)
        loss, sub_loss = yolov2_loss(detector_mask, matching_true_boxes, class_one_hot, true_boxes, y_pred)
    return loss, sub_loss, tape.gradient(loss, model.trainable_variables)

# save weights
def save_best_weights(model, name, val_loss_avg):
    # delete existing weights file
    files = glob.glob(os.path.join('weights/', name + '*'))
    for file in files:
        os.remove(file)
    # create new weights file
    name = name + '_' + str(val_loss_avg) + '.h5'
    path_name = os.path.join('weights/', name)
    model.save_weights(path_name)

# log (tensorboard)
def log_loss(loss, val_loss, step):
    tf.summary.scalar('loss', loss, step)
    tf.summary.scalar('val_loss', val_loss, step)


def train(epochs, model, train_dataset, val_dataset, steps_per_epoch_train, steps_per_epoch_val, train_name='train'):
    '''
    Train YOLO model for n epochs.
    Eval loss on training and validation dataset.
    Log training loss and validation loss for tensorboard.
    Save best weights during training (according to validation loss).

    Parameters
    ----------
    - epochs : integer, number of epochs to train the model.
    - model : YOLO model.
    - train_dataset : YOLO ground truth and image generator from training dataset.
    - val_dataset : YOLO ground truth and image generator from validation dataset.
    - steps_per_epoch_train : integer, number of batch to complete one epoch for train_dataset.
    - steps_per_epoch_val : integer, number of batch to complete one epoch for val_dataset.
    - train_name : string, training name used to log loss and save weights.

    Notes :
    - train_dataset and val_dataset generate YOLO ground truth tensors : detector_mask,
      matching_true_boxes, class_one_hot, true_boxes_grid. Shape of these tensors (batch size, tensor shape).
    - steps per epoch = number of images in dataset // batch size of dataset

    Returns
    -------
    - loss history : [train_loss_history, val_loss_history] : list of average loss for each epoch.
    '''
    num_epochs = epochs
    steps_per_epoch_train = steps_per_epoch_train
    steps_per_epoch_val = steps_per_epoch_val
    train_loss_history = []
    val_loss_history = []
    best_val_loss = 1e6

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # log (tensorboard)
    summary_writer = tf.summary.create_file_writer(os.path.join('logs/', train_name), flush_millis=20000)
    summary_writer.set_as_default()

    # training
    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_val_loss = []
        epoch_val_sub_loss = []
        print('Epoch {} :'.format(epoch))
        # train
        for batch_idx in range(steps_per_epoch_train):
            img, detector_mask, matching_true_boxes, class_one_hot, true_boxes = next(train_dataset)
            loss, _, grads = grad(model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(loss)
            print('-', end='')
        print(' | ', end='')
        # val
        for batch_idx in range(steps_per_epoch_val):
            img, detector_mask, matching_true_boxes, class_one_hot, true_boxes = next(val_dataset)
            loss, sub_loss, grads = grad(model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes,
                                         training=False)
            epoch_val_loss.append(loss)
            epoch_val_sub_loss.append(sub_loss)
            print('-', end='')

        loss_avg = np.mean(np.array(epoch_loss))
        val_loss_avg = np.mean(np.array(epoch_val_loss))
        sub_loss_avg = np.mean(np.array(epoch_val_sub_loss), axis=0)
        train_loss_history.append(loss_avg)
        val_loss_history.append(val_loss_avg)

        # log
        log_loss(loss_avg, val_loss_avg, epoch)

        # save
        if val_loss_avg < best_val_loss:
            save_best_weights(model, train_name, val_loss_avg)
            best_val_loss = val_loss_avg

        print(' loss = {:.4f}, val_loss = {:.4f} (conf={:.4f}, class={:.4f}, coords={:.4f})'.format(
            loss_avg, val_loss_avg, sub_loss_avg[0], sub_loss_avg[1], sub_loss_avg[2]))

    return [train_loss_history, val_loss_history]

def process(x, xcoord, ycoord, imgsize, grid_size, box_size):
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

    ycoord, xcoord = np.unravel_index(Image_heatmap.argmax(), Image_heatmap.shape)
    # Converting bbox to yolo format
    bbox_yolo = np.zeros((grid_size[0], grid_size[1], 5))
    grid_x = int(xcoord // (imgsize / grid_size[0]))
    grid_y = int(ycoord // (imgsize / grid_size[1]))
    bbox_yolo[grid_y, grid_x, 0] = (xcoord % (imgsize / grid_size[0])) / (imgsize / grid_size[0])
    bbox_yolo[grid_y, grid_x, 1] = (ycoord % (imgsize / grid_size[1])) / (imgsize / grid_size[1])
    bbox_yolo[grid_y, grid_x, 2] = box_size / imgsize
    bbox_yolo[grid_y, grid_x, 3] = box_size / imgsize
    bbox_yolo[grid_y, grid_x, 4] = 1

    return img, bbox_yolo

def get_gens(img_shape, paths, test_paths, batch_size, val_size, grid_size, box_size, anchors_count, class_count, max_annot):
    paths_idx = np.random.permutation(np.arange(len(paths)))
    thres = int(len(paths)*(1-val_size))
    tp, vp = [paths[x] for x in paths_idx[:thres]], [paths[x] for x in paths_idx[thres:]]
    return (fundus_gen(tp, batch_size, img_shape, grid_size, box_size, anchors_count, class_count, max_annot),
            fundus_gen(vp, batch_size, img_shape, grid_size, box_size, anchors_count, class_count, max_annot),
            fundus_gen(test_paths, batch_size, img_shape, grid_size, box_size, anchors_count, class_count, max_annot))


def fundus_gen(paths, batch_size, img_size, grid_size, box_size, anchors_count, class_count, max_annot):
    while True:
        batch_paths = [paths[i] for i in np.random.choice(a=np.arange(len(paths)), size=batch_size)]

        batch_imgs = []
        batch_detector_mask = []
        batch_matching_true_boxes = []
        batch_class_one_hot = []
        batch_true_boxes_grid = []

        for tup in batch_paths:
            img_path = tup[0]
            foveaCoords = tup[1]
            discCoords = tup[2]

            # Process image and get bounding box for fovea
            img, fovea_box = process(img_path, int(foveaCoords[0]), int(foveaCoords[1]), img_size, grid_size, box_size)
            # Process image and get bounding box for disc
            _, disc_box = process(img_path, int(discCoords[0]), int(discCoords[1]), img_size, grid_size, box_size)

            if len(img) > 0:
                batch_imgs.append(img)

                detector_mask = np.zeros((grid_size[0], grid_size[1], anchors_count, 1))
                detector_mask[..., 0] = (fovea_box[..., 4:5] > 0)
                batch_detector_mask.append(detector_mask)

                matching_true_boxes = np.zeros((grid_size[0], grid_size[1], anchors_count, 5))
                matching_true_boxes[..., 0] = fovea_box
                batch_matching_true_boxes.append(matching_true_boxes)

                class_one_hot = np.zeros((grid_size[0], grid_size[1], anchors_count, class_count))
                # Assuming class index 0 for fovea and 1 for disc
                class_one_hot[..., 0] = (fovea_box[..., 4:5] > 0)
                class_one_hot[..., 1] = (disc_box[..., 4:5] > 0)
                batch_class_one_hot.append(class_one_hot)

                true_boxes_grid = np.zeros((max_annot, 5))
                true_boxes_grid[0, :] = fovea_box.flatten()[:5]  # Using only x, y, w, h, c
                batch_true_boxes_grid.append(true_boxes_grid)

        yield [np.array(batch_imgs),
               np.array(batch_detector_mask),
               np.array(batch_matching_true_boxes),
               np.array(batch_class_one_hot),
               np.array(batch_true_boxes_grid)]
