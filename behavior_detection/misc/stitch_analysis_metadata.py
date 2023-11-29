import os
import glob

import pandas as pd



def stitch_tracklets(batches_folder: str):
    batches = os.listdir(batches_folder)
    for i in range(len(batches)):
        curr_dir = os.path.join(batches_folder, batches[i])
        try:
            hdf = pd.read_hdf(glob.glob(os.path.join(curr_dir, "*filtered.h5"))[0])
            csv = pd.read_csv(glob.glob(os.path.join(curr_dir, "*filtered.csv"))[0], header=[0, 1, 2, 3])
            #print(hdf.keys())
            print(csv)
        except FileNotFoundError as e:
            print(e)

if __name__ == "__main__":
    batches = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc36_2_Tk3_030320/batches"

    stitch_tracklets(batches)
