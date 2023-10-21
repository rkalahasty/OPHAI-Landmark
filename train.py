#train.py

import argparse

parser = argparse.ArgumentParser(description='Train Landmark model for Project SegLoc (OPHAI).')

parser.add_argument('--model_name', '--model_name', help='Name of model to train.',
                    choices=['unet', 'vggunet', 'resunet', 'hbaunet', 'swinunet', 'hbaunet+attnet', 'unetplusplus',
                             'unetplusplusDE', 'deeplab', 'pix2pix', 'attnet', 'detectron2', "yolov2", "google_model"],
                    required=True)
parser.add_argument('--tr', '--train', help='Name of the CSV file with training dataset information.', required=True)
parser.add_argument('--te', '--test', help='Name of the CSV file with testing dataset information.', required=False)
parser.add_argument('--dd', '--dataset_dir', help='Path to the folder with the CSV files and image subfolders.',
                    required=True)
parser.add_argument('--sp', '--save_path',
                    help='Path to the folder where trained models and all metrics/graphs will be saved.', required=True)

parser.add_argument('--img', '--image_size', type=int,
                    help='Size to which the images should be reshaped (one number, i.e. 256 or 512).', required=True)
parser.add_argument('--multiprocessing', '--multiprocessing', type=int, help='Amount of Threads to Use')
args = parser.parse_args()

import os
import numpy as np
import tensorflow as tf
import multiprocessing
from tqdm import tqdm
from PIL import Image
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
import sys
sys.path.insert(0, '/OPHAI-Landmark-Localization/ophai')
sys.path.append('../ophai/')
import pandas as pd
import tensorflow_addons as tfa

print(tf.__version__)

train_path = os.path.join(args.dd, args.tr)
test_path = os.path.join(args.dd, args.te)
img_size = (args.img, args.img)

dataset_dir = args.dd

df_train = pd.read_csv(train_path)[['imageID', 'imageDIR', 'Fovea_X', 'Fovea_Y', 'Disc_X', 'Disc_Y']].values.tolist()
train_paths = []

for r in df_train:
    img_path = os.path.join(os.path.split(dataset_dir)[0], r[1], r[0])
    
    train_paths.append((img_path, (r[2], r[3]), (r[4], r[5])))

df_test = pd.read_csv(test_path)[['imageID', 'imageDIR', 'Fovea_X', 'Fovea_Y', 'Disc_X', 'Disc_Y']].values.tolist()
test_paths = []

for r in df_test:
    img_path = os.path.join(os.path.split(dataset_dir)[0], r[1], r[0])
    
    test_paths.append((img_path, (r[2], r[3]), (r[4], r[5])))

from models.unet import Unet
from models.resunet import ResUnet
from models.vggunet import Vggunet
from models.swinunet import swinunet
from models.hbaunet import hbaunet
from models.uneteffnet import uneteffnet
from models.deeplab import deeplab
from models.attnet import attentionunet
from models.unet2plus import Unet2
from datautils import process, get_gens
from trainyolo import trainYOLO
from models.yolov2 import YoloV2
from models.pix2pix import *
# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultTrainer

def get_dataset(df, shuffle=False):
    img_paths = df['imageDIR'].values
    labels = df['label'].values

    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_paths))
    ds = ds.map(lambda x, y: (load_image(x), y)) 
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds

if args.model_name == "swinunet":
    model = swinunet((args.img, args.img, 3), filter_num_begin=16, n_labels=2, depth=6, stack_num_down=4, stack_num_up=4,
                     patch_size=(2, 2), num_heads=[4, 8, 16, 16, 32, 32], window_size=[4, 4, 2, 2, 2, 2], num_mlp=4,
                     output_activation='Sigmoid', shift_window=True,
                     name='swin_unet')
elif args.model_name == "unetplusplus":
    model = Unet2(args.img, args.img, color_type=3, num_class=2, deep_supervision=False)
elif args.model_name == "unetplusplusDE":
    model = Unet2(args.img, args.img, color_type=3, num_class=2, deep_supervision=True)
elif args.model_name == "attnet":
    model = attentionunet((args.img, args.img, 3))
elif args.model_name == "hbaunet":
    model = hbaunet((args.img, args.img, 3), dropout_rate=0.4, use_attnDecoder=False, skip=False, num_layers=3)
elif args.model_name == "hbaunet+attnet":
    model = hbaunet((args.img, args.img, 3), dropout_rate=0.4, use_attnDecoder=True, skip=False, num_layers=3)
elif args.model_name == "yolov2":
    model = YoloV2(input_shape=(256, 256, 3),  
               GRID_H=8, GRID_W=8)
elif args.model_name == "google_model":
    df_train = pd.DataFrame(df_train, columns=['imageID', 'imageDIR', 'Fovea_X', 'Fovea_Y', 'Disc_X', 'Disc_Y']) 
    
    @tf.function
    def get_patches(images):
        return tf.image.extract_patches(images, sizes=[1, patch_size, patch_size, 1], strides=[1, patch_size, patch_size, 1], rates=[1, 1, 1, 1], padding='VALID')


    # Split validation set
    val_size = 0.2 
    val_idx = int(len(df_train) * (1-val_size))
    val_df = df_train.iloc[val_idx:]
    df_train = df_train.iloc[:val_idx]
    batch_size = 8
    train_gen, val_gen, _ = get_gens(args.img, train_paths, [], batch_size, val_size=val_size)

    from models_vit import *
    import tfimm

    from scheduler import WarmUpCosine

    # Load model
    model = tfimm.create_model( # apply global pooling without class token
        "vit_large_patch16_224_mae",
        nb_classes = 2
    )

    # Load pretrained weights 
    model.load_weights('RETFound_cfp_weights.h5', by_name=True, skip_mismatch=True)

    inputs = tf.keras.Input(shape=(256, 256, 3))

    # Patch extraction
    patch_size = 16
    patches = get_patches(images=inputs)

    # Interpolation function
    def interpolate_pos_embedding(pos_embed, seq_len):
        # Resize pos embed  
        pos_embed = tf.cast(pos_embed, tf.float32)
        new_pos_embed = tf.image.resize(pos_embed, size=(seq_len, pos_embed.shape[2]))
        return new_pos_embed

    # Interpolate 
    seq_len = patches.shape[1] 
    pos_embed = interpolate_pos_embedding(model.pos_embed, seq_len)

    # Add position embeddings 
    x = patches + pos_embed

    _ = model(x)

    lr = 1e-4
    warmup_epochs = 30
    total_epochs = 300

    # Dataset size
    num_samples = len(train_paths) 

    # Batch size 
    batch_size = 64

    # Steps per epoch
    steps_per_epoch = num_samples // batch_size

    # Total steps 
    total_epochs = 100
    total_steps = total_epochs * steps_per_epoch

    # Warmup
    warmup_epoch_percentage = 10
    warmup_steps = total_steps * warmup_epoch_percentage / 100 


    scheduled_lrs = WarmUpCosine(lr, total_steps, warmup_epochs, warmup_steps)

    # Compile    
    optimizer = tfa.optimizers.AdamW(
        learning_rate=scheduled_lrs,
        weight_decay=1e-4  
    )

    loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # Add sigmoid layer for conversion to probability
    x = model.output
    x = tf.keras.layers.Dense(2, activation='sigmoid', kernel_initializer='glorot_uniform')(x)

    outputs = x
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        'vit_model_{epoch:03d}.h5', 
        save_weights_only=True
    ) 

    # Callback for early stopping  
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5
    )

    # Define callbacks list
    callbacks = [
        checkpoint_cb,
        early_stopping_cb
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    # Train
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=300,
        callbacks=callbacks
    )


elif args.model_name == "detectron2":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("landmark_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = args.multiprocessing
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 200
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = args.sp + '\checkpoints'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
elif args.model_name == "pix2pix":
    # Build the models
    generator = build_generator()
    discriminator = build_discriminator()

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

else:
    model = {
        'unet': Unet,
        'resunet': ResUnet,
        'vggunet': Vggunet,
        'swinunet': swinunet,
        'deeplab': deeplab,
    }[args.model_name]((args.img, args.img, 3), 2)
if args.model_name == 'detectron2':
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
elif args.model_name == "pix2pix":
    batch_size = 8
    val_size = 0.1

    train_gen, val_gen, _ = get_gens(args.img, train_paths, [], batch_size, val_size=val_size)
    train_len = int(len(train_paths) * (1 - val_size))
    val_len = len(train_paths) - train_len
    fit(train_gen, val_gen, train_len // batch_size, val_len // batch_size, 300, '{}_generator_model.h5')
elif args.model_name=="yolov2":
    trainYOLO(train_paths, test_paths, args.img)
else:
    os.chdir(args.sp)
    val_size = 0.1
    batch_size = 8
    input_shape = (args.img, args.img, 3)

    train_gen, val_gen, _ = get_gens(args.img, train_paths, [], batch_size, val_size=val_size)
    train_len = int(len(train_paths) * (1 - val_size))
    val_len = len(train_paths) - train_len

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_absolute_error',
                      metrics=['mean_absolute_error', 'accuracy'])
    model.summary()

    from tensorflow.keras.callbacks import EarlyStopping

    checkpoint_filepath = '/' + args.sp + '/' + args.model_name + '_checkpoint'

    os.chdir(args.sp + '\checkpoints')


    filepath = args.model_name+'epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
    from keras.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                verbose = 1,
                                 save_best_only = True,
                                    mode ='min')


    es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
    history = model.fit(train_gen, validation_data = val_gen, steps_per_epoch=train_len//batch_size, validation_steps = val_len//batch_size, callbacks=[es, checkpoint], verbose = 1, epochs = 300)

    os.chdir(args.sp)

    model.save(args.model_name + ".h5")

    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{args.model_name} Loss During Training')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.savefig(f'{args.model_name}loss.png', bbox_inches='tight')

    np.save(f'{args.model_name}_TrainingHistory.npy', history.history)