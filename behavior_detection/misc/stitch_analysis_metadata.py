import os
import glob

import pandas as pd
import numpy as np


def get_difference(a: pd.Series, b: pd.Series):
    fish_a = a.keys()[0][0]
    fish_b = b.keys()[0][0]
    a = a.droplevel(0)
    b = b.droplevel(0)
    diff = np.sum(np.abs(a - b))
    return diff


def get_pairs(a: pd.Series, b: pd.Series):
    """
    Links fish in the first row with fish in the second row based on position similarity
    """
    pairs = {}
    # loop through b (b has no gaps, so we can stop once we hit an empty fish slot
    for i in range(1, b.shape[0], 27):
        b_fish: pd.Series = b.iloc[i:i + 27]
        if b_fish.isna().sum() == 27:  # no fish in this slot
            break
        for j in range(1, a.shape[0], 27):
            a_fish: pd.Series = a.iloc[j:j + 27]
            a_label = a_fish.keys()[0][0]

            if pairs.get(a_label):  # fish already linked
                continue
            if a_fish.isna().sum() == 27:  # no fish in this slot, move onto next slot
                continue

            if get_difference(a_fish, b_fish) >= 200:  # positions are not similar enough
                continue

            pairs.update({a_label: b_fish.keys()[0][0]})

    return pairs


def stitch_tracklets(batches_folder: str):
    batches = sorted(os.listdir(batches_folder))
    for i in range(1, len(batches)):
        prev_dir = os.path.join(batches_folder, batches[i - 1])
        curr_dir = os.path.join(batches_folder, batches[i])
        try:
            hdf = pd.read_hdf(glob.glob(os.path.join(curr_dir, "*filtered.h5"))[0])
            prev_csv = pd.read_csv(glob.glob(os.path.join(prev_dir, "*filtered.csv"))[0], header=[0, 1, 2, 3])
            curr_csv = pd.read_csv(glob.glob(os.path.join(curr_dir, "*filtered.csv"))[0], header=[0, 1, 2, 3])
            prev_csv.columns = prev_csv.columns.droplevel(0)
            curr_csv.columns = curr_csv.columns.droplevel(0)
            table = get_pairs(prev_csv.iloc[-1], curr_csv.iloc[0])
            print(f"prev: batch{i - 1}, curr: batch{i}. links: {table}")
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    batches = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc36_2_Tk3_030320/batches"
    stitch_tracklets(batches)
