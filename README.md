# OPHAI Landmarking

This study aims to create a comprehensive benchmark for the evaluation of deep learning methods for the localization of the fovea and disc in fundus images under various perturbations. Such a benchmark is essential for the development of new landmark localization models that can work in different real imaging conditions including brightness, contrast, lens effects, etc.

This repository contains the code used for the ARVO abstract published here: https://iovs.arvojournals.org/article.aspx?articleid=2791055, from OPHAI Lab (supervised by Dr. Mohammad Eslami, Dr. Saber Kazeminasab, and Dr. Tobias Elze). Instructions for reproducing results and using this repository are below. All files for training, evaluating, and plotting are located in the parent directory. The code for all models we used are in `/models`.

![](Perturbation.png)

## Obtaining Datasets

Contact one of the member of Dr. Eslami's lab and they will share the dataset to you. When downloading the dataset, put it at the same level as the repository for the code in your directory. If you don't it will lead to many errors since the paths have to be modified.

## Installation Process

Follow the instructions that's corresponding to your system.

### Windows (Recommended System)

### MacOS

1. Install Anaconda
2. Open Anaconda-Navigator
3. On the left side, open environments
4. Go to the bottom and hit "Create"
5. Put a name for the environment (preferrably "OPHAI")
6. Set the Python version to around `3.9`
7. Open the repository in your IDE and activate the environment using `conda activate OPHAI`
8. Install the necessary packages using `pip3 install -r mac-requirements.txt` or `pip install -r requirements.txt`

## Using the Files

**NOTE**: If you are using a Mac, replace `python` with `python3` in the commands. Additionally, training models on a Macbook is not recommended since it may take a long time for the models to be properly trained.

`ROOT_DIRECTORY` refers to the area in which you stored the files. (e.g., C:\Users\abc\Desktop)

### Training models

For your convenience, we provided the scripts in folders chracterized by the deep learning method used. The Train files have multiple arguements, some of which are optional. An important thing to note is that the train files allow the option for you to load in numpy arrays to train/test the model on. If you do not load in your own numpy arrays, the file saves the preprocessed train and test arrays onto your computer so you wont need to go through the whole preprocessing process again, saving you time.

Example Usage:

```
python train.py
  --model_name yolov2
  --tr data_FullFundus_256_loc_joint_train.csv
  --te data_FullFundus_256_loc_joint_test.csv
  --dd ROOT_DIRECTORY/OPHAIResults/newTrainCSVfiles
  --sp ROOT_DIRECTORY/OPHAIResults/newModelResults
  --img 256
```

`--te` is an optional tag that you can add. It is only needed by the YOLO v2 model as of September 17, 2023.

The model names that can be provided are below:

- hbaunet
- yolov2
- swinunet
- unetplusplus
- unetplusplusDE
- attnet
- hbaunet+attnet
- detectron2

The files that contain the code for the models can be found in `/models`.

### Generating Metrics

A singular testing file can be used to get results from every single model, you have to specify a path to which trained model you want to use. The following usage helps generate metrics for the models to determine which model would be the best for landmarking.

Usage:

```
python testp2.py
  --tr ROOT_DIRECTORY/OPHAIresults/NewTrainCSVfiles/data_FullFundus_256_loc
  --dd ROOT_DIRECTORY/OPHAIresults/NewTrainCSVfiles
  --path_trained_model ROOT_DIRECTORY/OPHAI-Landmark-Localization/SAVED_WEIGHT_FILE
  --sp ROOT_DIRECTORY/OPHAIresults/modelResults
  --img 256
```

### Running Predictions

All of the files that can be used to create predictions are in `/predictions`. If you need to generate a new file to make predictions, do so by creating a new file in that folder.
