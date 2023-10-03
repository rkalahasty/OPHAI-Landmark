import argparse

parser = argparse.ArgumentParser(description='Train Landmark model for Project SegLoc (OPHAI).')

parser.add_argument('--model_name', '--model_name', help='Name of model to train.',
                    choices=['unet', 'vggunet', 'resunet', 'hbaunet', 'swinunet', 'hbaunet+attnet', 'unetplusplus',
                             'unetplusplusDE', 'deeplab', 'pix2pix', 'attnet', 'detectron2'],
                    required=True)
parser.add_argument('--tr', '--train', help='Name of the CSV file with training dataset information.', required=True)
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
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
import sys
sys.path.insert(0, '/OPHAI-Landmark-Localization-main/ophai')
sys.path.append('../ophai/')
import pandas as pd

print(tf.__version__)

train_path = os.path.join(args.dd, args.tr)
img_size = (args.img, args.img)

dataset_dir = args.dd

df_train = pd.read_csv(train_path)[['imageID', 'imageDIR', 'Fovea_X', 'Fovea_Y', 'Disc_X', 'Disc_Y']].values.tolist()
train_paths = []
for r in df_train:
    img_path = os.path.join(os.path.split(dataset_dir)[0], r[1], r[0])
    train_paths.append((img_path, (r[2], r[3]), (r[4], r[5])))

from models.unet import Unet
from models.resunet import ResUnet
from models.vggunet import Vggunet
from models.swinunet import swinunet
from models.hbaunet import hbaunet
from models.deeplab import deeplab
from models.attnet import attentionunet
from models.unet2plus import Unet2
from datautils import get_gens
from models.pix2pix import *
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import os, random, matplotlib.pyplot as plt
import cv2

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
elif args.model_name == "detectron2":

    from detectron2.data import DatasetCatalog, MetadataCatalog
    from datautils import detectron_gen, get_train

    tp = get_train(train_paths)
    for d in [("train", tp)]:
        DatasetCatalog.register("fundusLandmarking_" + d[0], lambda d=d: detectron_gen(d[1], args.img))
        MetadataCatalog.get("fundusLandmarking_" + d[0]).set(thing_classes=["Fovea", "Disk"])
    metadata = MetadataCatalog.get("fundusLandmarking_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("fundusLandmarking_train")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.BACKBONE.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER =  1000  # adjust according to your dataset size and needs
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.OUTPUT_DIR = args.sp + '\yaml_checkpoints'
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
    from tensorflow.keras.callbacks import ModelCheckpoint

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