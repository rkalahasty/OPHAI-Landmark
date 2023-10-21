import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import  tqdm
import os
import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)


direc = r"C:\Users\17033\Desktop\p2OPHAIresults\modelResults"
os.chdir(direc)
orig = direc + r"\hbaunet.h5_orig_test.csvResults.csv"
d1 = direc + r"\hbaunet.h5_d1_test.csvResults.csv"
d2 = direc + r"\hbaunet.h5_d2_test.csvResults.csv"

def rmse(predictions, targets):
    return mean_squared_error(predictions, targets, squared=False)

#ORIG
def getOrigData(file):
    EdData = pd.read_csv(file)[['ED_Fovea', 'ED_Disc']].values.tolist()
    df = pd.DataFrame(columns=['Avg_ED_Fovea','Avg_ED_Disk', 'Stddev_Fovea', 'Stddev_Disk'])
    Avg_ED_Fovea = sum(i[0] for i in EdData)/len(EdData)
    Avg_ED_Disk = sum(i[1] for i in EdData)/len(EdData)
    Stddev_Fovea = np.std([i[0] for i in EdData])
    Stddev_Disk = np.std([i[1] for i in EdData])
    df.to_csv(f'{file}_Compact_ED.csv', index=False)

    df = df.append({'Avg_ED_Fovea': Avg_ED_Fovea , 'Avg_ED_Disk': Avg_ED_Disk, 'Stddev_Fovea': Stddev_Fovea, 'Stddev_Disk': Stddev_Disk}, ignore_index=True)

    df.to_csv(f'{file}_Compact_ED.csv', index=False)

#D1 Files

def getD1Data(file):
    EdData = pd.read_csv(file)[['img_path', 'ED_Fovea', 'ED_Disc']].values.tolist()
    df = pd.DataFrame(columns=['De-illumination', 'De-spot', 'De-illumination + De-spot'])

    PredData = pd.read_csv(file)[['gt_Fovea_X', 'gt_Fovea_Y', 'gt_Disc_X', 'gt_Disc_Y', 'p_Fovea_X', 'p_Fovea_Y', 'p_Disc_X', 'p_Disc_Y']].values.tolist()

    dataPerturb = [[], [], []]

    for i in tqdm(range(len(EdData))):
        path = EdData[i][0]
        gt_Fovea_X, gt_Fovea_Y, gt_Disc_X, gt_Disc_Y, p_Fovea_X, p_Fovea_Y, p_Disc_X, p_Disc_Y = PredData[i]
        hba_unet_foveaED = EdData[i][1]; hba_unet_diskED = EdData[i][2]

        path = os.path.basename(os.path.normpath(path))
        path = path.split('_')
        preturbation = path[3]
        if preturbation == "001": dataPerturb[0].append(((hba_unet_foveaED + hba_unet_diskED)/2, hba_unet_foveaED, hba_unet_diskED, gt_Fovea_X, gt_Fovea_Y, gt_Disc_X, gt_Disc_Y, p_Fovea_X, p_Fovea_Y, p_Disc_X, p_Disc_Y ) )
        elif preturbation == "010":  dataPerturb[1].append( ((hba_unet_foveaED + hba_unet_diskED)/2, hba_unet_foveaED, hba_unet_diskED, gt_Fovea_X, gt_Fovea_Y, gt_Disc_X, gt_Disc_Y, p_Fovea_X, p_Fovea_Y, p_Disc_X, p_Disc_Y ) )
        else: dataPerturb[2].append( ((hba_unet_foveaED + hba_unet_diskED)/2, hba_unet_foveaED, hba_unet_diskED, gt_Fovea_X, gt_Fovea_Y, gt_Disc_X, gt_Disc_Y, p_Fovea_X, p_Fovea_Y, p_Disc_X, p_Disc_Y ) )


    ED001, ED010, ED011 = sum(i[0] for i in dataPerturb[0])/len(dataPerturb[0]), sum(i[0] for i in dataPerturb[1])/len(dataPerturb[0]), sum(i[0] for i in dataPerturb[2])/len(dataPerturb[0])
    EDF001, EDF010, EDF011 = sum(i[1] for i in dataPerturb[0])/len(dataPerturb[0]), sum(i[1] for i in dataPerturb[1])/len(dataPerturb[0]), sum(i[1] for i in dataPerturb[2])/len(dataPerturb[0])
    EDD001, EDD010, EDD011 = sum(i[2] for i in dataPerturb[0])/len(dataPerturb[0]), sum(i[2] for i in dataPerturb[1])/len(dataPerturb[0]), sum(i[2] for i in dataPerturb[2])/len(dataPerturb[0])
    df = df.append({'De-illumination' : ED001, 'De-spot': ED010, 'De-illumination + De-spot': ED011}, ignore_index=True)
    df = df.append({'De-illumination' : EDF001, 'De-spot': EDF010, 'De-illumination + De-spot': EDF011}, ignore_index=True)
    df = df.append({'De-illumination' : EDD001, 'De-spot': EDD010, 'De-illumination + De-spot': EDD011}, ignore_index=True)


    std001, std010, std011 = np.std([i[0] for i in dataPerturb[0]]), np.std([i[0] for i in dataPerturb[1]]), np.std([i[0] for i in dataPerturb[2]])
    stdF001, stdF010, stdF011 = np.std([i[1] for i in dataPerturb[0]]), np.std([i[1] for i in dataPerturb[1]]), np.std([i[1] for i in dataPerturb[2]])
    std001, stdD010, stdD011 = np.std([i[2] for i in dataPerturb[0]]), np.std([i[2] for i in dataPerturb[1]]), np.std([i[2] for i in dataPerturb[2]])

    df = df.append({'De-illumination' : std001, 'De-spot': std010, 'De-illumination + De-spot': std011}, ignore_index=True)
    df = df.append({'De-illumination' : stdF001, 'De-spot': stdF010, 'De-illumination + De-spot': stdF011}, ignore_index=True)
    df = df.append({'De-illumination' : std001, 'De-spot': stdD010, 'De-illumination + De-spot': stdD011}, ignore_index=True)

    df.to_csv(f'{file}_Compact_ED.csv', index=False)

def getD2Data(file):
    EdData = pd.read_csv(file)[['img_path', 'ED_Fovea', 'ED_Disc']].values.tolist()
    df = pd.DataFrame(columns=['Perturbation', 'EDFovea', "EDDisc"])

    for i in tqdm(range(len(EdData))):
        path = EdData[i][0]
        path = os.path.basename(os.path.normpath(path))
        for k in ['gaussian_noise', 'Hue_minus', 'Hue_plus', 'impulse_noise', 'jpeg_compression', 'Saturation_minus',
                  'Saturation_plus', 'shot_noise', 'speckle_noise']:
            if k in path:
                path = path.replace(k, k.replace("_", ""))
                break

        path = path.split('_')
        Perturbation = path[3]
        severity = path[4]
        severity = severity[0]
        if Perturbation != "Temperature" and severity == "3": df = df.append({'Perturbation': Perturbation, 'EDFovea' :EdData[i][1], 'EDDisc': EdData[i][2]}, ignore_index=True)

    perturbDct = {

    }
    print(df.Perturbation.unique())
    for i in df.Perturbation.unique():
        perturbDct[i] = [[], []]

    for ind, i in df.iterrows():
        x = print(i[0])
        print(i[1])
        print(i[2])

        perturbDct[i['Perturbation']][0].append(i['EDFovea'])
        perturbDct[i['Perturbation']][1].append(i['EDDisc'])

    dfd2 = pd.DataFrame(columns=df.Perturbation.unique())
    F_edRow = {}
    D_edRow = {}
    F_stdRow = {}
    D_stdRow = {}

    for i in df.Perturbation.unique():
        EDFoveaList = perturbDct[i][0]
        EDDiscList = perturbDct[i][1]
        F_edRow[i] = sum(EDFoveaList)/len(EDFoveaList)
        D_edRow[i] = sum(EDDiscList)/len(EDDiscList)
        F_stdRow[i] = np.std(EDFoveaList)
        D_stdRow[i] = np.std(EDDiscList)
    dfd2 = dfd2.append(F_edRow, ignore_index=True)
    dfd2 = dfd2.append(D_edRow, ignore_index=True)
    dfd2 = dfd2.append(F_stdRow, ignore_index=True)
    dfd2 = dfd2.append(D_stdRow, ignore_index=True)

    dfd2.to_csv(f'{file}_Compact_ED.csv', index=False)

getOrigData(orig)
getD1Data(d1)
getD2Data(d2)