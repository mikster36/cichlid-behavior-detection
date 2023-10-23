import pickle
import pandas as pd
import numpy as np

filepath_csv = r"/home/bree_student/Downloads/MasterAnalysisFiles/AllDetectionsFish.csv"
filepath_pickle = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_assemblies.pickle"
filepath_h5 = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_el.h5"

data_csv = pd.read_csv(filepath_csv)
data_pickle = pd.read_pickle(filepath_pickle)
data_h5 = pd.read_hdf(filepath_h5)

item = data_pickle[0][0]
print(item)
"""
for key, value in data_pickle.items():
    print(value[0])
"""