import pickle
import pandas as pd

filepath_ = "/home/bree_student/Downloads/dlc_model-student-2023-07-26/dlc-output/iteration-1/52.389_55.3661DLC_dlcrnetms5_dlc_modelJul26shuffle1_200000_assemblies.pickle"

df = pd.read_pickle(filepath_)

print(df)