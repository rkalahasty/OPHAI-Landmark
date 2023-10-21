"""
Created on Sun Jan  1 23:33:56 2023

@author: Moha-Cate
"""

import pandas
import numpy as np

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats as st
import pandas as pd
from tqdm import tqdm
dir = r"C:\Users\17033\Desktop\p2OPHAIresults\modelResults"
os.chdir(dir)

list_of_Results = [
    ('dlib', 'orig', r'\',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_orig.csv'),
    ('dlib', 'd1', '../results/unetpp/d1/individual_results.csv',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d1.csv'),
    ('dlib', 'd2', '../results/unetpp/d2/individual_results.csv',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d2.csv'),

    ('DeepLabV3+', 'orig', '../results/deeplabv3plus/orig/individual_results.csv',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_orig.csv'),
    ('DeepLabV3+', 'd1', '../results/deeplabv3plus/d1/individual_results.csv',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d1.csv'),
    ('DeepLabV3+', 'd2', '../results/deeplabv3plus/d2/individual_results.csv',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d2.csv'),

    ('CE-Net', 'orig', '../results/cenet/orig/individual_results.csv',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_orig.csv'),
    ('CE-Net', 'd1', '../results/cenet/d1/individual_results.csv',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d1.csv'),
    ('CE-Net', 'd2', '../results/cenet/d2/individual_results.csv',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d2.csv'),

    ('CE-Net', 'orig', '../results/cenet/orig/individual_results.csv',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_orig.csv'),
    ('CE-Net', 'd1', '../results/cenet/d1/individual_results.csv',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d1.csv'),
    ('CE-Net', 'd2', '../results/cenet/d2/individual_results.csv',
     '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d2.csv'),
]

f = open('resultsFile', 'w')
