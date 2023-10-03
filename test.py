import argparse

parser = argparse.ArgumentParser(description='Testing Landmark Localization model for Project SegLoc (OPHAI).')

parser.add_argument('--tr', '--test', help='Name of the CSV file with testing dataset information.', required=True)
parser.add_argument('--dd', '--dataset_dir', help='Path to the folder with the CSV files and image subfolders.',
                    required=True)
parser.add_argument('--model_type', '--model_type',
                    choices=['unet', 'vggunet', 'resunet', 'hbaunet', 'swinunet', 'hbaunet+attnet', 'unetplusplus',
                             'unetplusplusDE', 'deeplab', 'pix2pix', 'attnet', 'detectron2'],
                    required=True)

parser.add_argument('--path_trained_model', '--path_trained_model',
                    help='Path to the numpy array with the train files and image subfolders.',
                    required=False)
parser.add_argument('--sp', '--save_path',
                    help='Path to the folder where trained models and all metrics/graphs will be saved.',
                    required=True)
parser.add_argument('--img', '--image_size', type=int,
                    help='Size to which the images should be reshaped (one number, i.e. 256 or 512).', required=True)

args = parser.parse_args()

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datautils import process
import csv
from sklearn.metrics import mean_squared_error
from models.hbaunet import hbaunet
from models.hbaunet import HBA
from detectron2.data import DatasetCatalog, MetadataCatalog
from datautils import detectron_gen, get_train

from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from datautils import detectron_gen, get_train
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
np_config.enable_numpy_behavior()

print(tf.__version__)

def bbox_center(bbox):
    """Compute the center of a bounding box."""
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def ED(x, gt):
    prediction = np.unravel_index(x.argmax(), x.shape)
    groundtruth = np.unravel_index(gt.argmax(), gt.shape)
    dist = np.linalg.norm(np.asarray([prediction[0], prediction[1]]) - np.asarray(groundtruth))
    return dist, prediction[0], prediction[1], groundtruth[0], groundtruth[1]


def rmse(predictions, targets):
    return mean_squared_error(predictions, targets, squared=False)

def RCNN_DAT(loc, gt):
    groundtruth = np.unravel_index(gt.argmax(), gt.shape)
    dist = np.linalg.norm(np.asarray([loc[0], loc[1]]) - np.asarray(groundtruth))
    return dist, loc[0], loc[1], groundtruth[0], groundtruth[1]


# GET PREDICTION BASED RESUlTS FOR BOXPLOTS

for i in ["_d2_test.csv", "_d1_test.csv",  "_orig_test.csv",  "_d4_test.csv", "_d5_test.csv"]:
    if args.model_type == 'detectron2':
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("fundusLandmarking_test")
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 0
        # cfg.MODEL.BACKBONE.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 1000  # adjust according to your dataset size and needs
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.OUTPUT_DIR = args.sp + '\yaml_checkpoints'
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set a custom testing threshold
        predictor = DefaultPredictor(cfg)
    elif args.model_type == "hbaunet+attnet":
        model = hbaunet((args.img, args.img, 3), dropout_rate=0.4, use_attnDecoder=True, skip=False, num_layers=3)
        model.load_weights(args.path_trained_model)
    elif args.model_type == "hbaunet":
        model = hbaunet((args.img, args.img, 3), dropout_rate=0.4, use_attnDecoder=False, skip=False, num_layers=3)
        model.load_weights(args.path_trained_model)

    else:
        model = tf.keras.models.load_model(args.path_trained_model)

    os.chdir(args.dd)

    train_path = args.tr + i
    df_test = pd.read_csv(train_path)[['imageID', 'imageDIR', 'Fovea_X', 'Fovea_Y', 'Disc_X', 'Disc_Y']].values.tolist()
    foveaEDbin = []
    diskEDbin = []
    FoveaPreds = []
    DiskPreds = []
    FoveaGts = []
    DiskGts = []

    out_write = open(args.path_trained_model + i + 'Results.csv', 'w')
    csvwriter = csv.writer(out_write)

    fields = ['img_path', 'gt_Fovea_X', 'gt_Fovea_Y', 'gt_Disc_X', 'gt_Disc_Y', 'p_Fovea_X', 'p_Fovea_Y', 'p_Disc_X',
              'p_Disc_Y', "ED_Fovea", "ED_Disc"]
    rows = []
    csvwriter.writerow(fields)

    os.chdir(args.sp)

    for r in tqdm(df_test):
        try:
            img_path = os.path.join(os.path.split(args.dd)[0], r[1], r[0])
            img, Fovea = process(img_path, int(r[2]), int(r[3]), args.img)
            img, Disk = process(img_path, int(r[4]), int(r[5]), args.img)
            if type(img) != list:
                if args.model_type == 'detectron2':
                    outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
                    pred_centers = [bbox_center(box) for box in pred_boxes]
                    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
                    pred_scores = outputs["instances"].scores.cpu().numpy()
                    max_score_indices = {}
                    for idx, score in enumerate(pred_scores):
                        if pred_classes[idx] not in max_score_indices or score > pred_scores[
                            max_score_indices[pred_classes[idx]]]:
                            max_score_indices[pred_classes[idx]] = idx

                    # Compute RMSE for each class using the highest accuracy prediction
                    for center, pt_class in zip(pred_centers, pred_classes):
                        if pt_class in max_score_indices:
                            pred_center = pred_centers[max_score_indices[pt_class]]

                            if pt_class == 0:  # Assuming 0 represents Fovea, 1 represents Disc
                                EDFovea, fpX, fpY, fgX, fgY = RCNN_DAT(pred_center, Fovea)
                            elif pt_class == 1:
                                EDDisk, dpX, dpY, dgX, dgY = RCNN_DAT(pred_center, Disk)

                else:
                    output = model(np.array([img]))
                    output = np.array(output[0])
                    EDFovea, fpX, fpY, fgX, fgY = ED(output[:, :, 0], Fovea)
                    EDDisk, dpX, dpY, dgX, dgY = ED(output[:, :, 1], Disk)

                row = [img_path, fgX, fgY, dgX, dgY, fpX, fpY, dpX, dpY, EDFovea, EDDisk]
                rows.append(row)
                foveaEDbin.append(EDFovea)
                diskEDbin.append(EDDisk)
                FoveaPreds.append((fpX, fpY))
                DiskPreds.append((dpX, dpY))
                FoveaGts.append((fgX, fgY))
                DiskGts.append((dgX, dgY))
        except FileNotFoundError:
            continue
        except RuntimeError:
            continue
        except OSError:
            continue
        except ValueError:
            continue
    csvwriter.writerows(rows)

    # GET COMPACT RESULTS
    compact_out_write = open(args.path_trained_model + i + 'CompactResults.csv', 'w', newline='')
    csvwriter = csv.writer(compact_out_write)
    fields = ['Avg_ED_Fovea', 'Avg_ED_Disk', 'RMSE_Fovea', 'RMSE_Disk']
    Avg_ED_Fovea = sum(foveaEDbin) / len(foveaEDbin)
    Avg_ED_Disk = sum(diskEDbin) / len(diskEDbin)
    RMSE_Fovea = rmse(FoveaPreds, FoveaGts)
    RMSE_Disk = rmse(DiskPreds, DiskGts)
    row = [str(Avg_ED_Fovea), str(Avg_ED_Disk), str(RMSE_Fovea), str(RMSE_Disk)]
    csvwriter.writerow(fields)
    csvwriter.writerow(row)

