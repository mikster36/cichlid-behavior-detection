import os
import glob

import pandas as pd
import numpy as np
from pathlib import Path

col_1 = ['individuals'] + [f'fish{i}' for i in range(1, 11) for _ in range(27)]
col_2 = ['bodyparts'] + ['nose', 'nose', 'nose', 'lefteye', 'lefteye', 'lefteye',
                         'righteye', 'righteye', 'righteye', 'spine1', 'spine1', 'spine1',
                         'spine2', 'spine2', 'spine2', 'spine3', 'spine3', 'spine3',
                         'backfin', 'backfin', 'backfin', 'leftfin', 'leftfin', 'leftfin',
                         'rightfin', 'rightfin', 'rightfin'] * 10
col_3 = ['coords'] + ['x', 'y', 'likelihood'] * 9 * 10
INDEX = pd.MultiIndex.from_arrays([col_1, col_2, col_3])


def get_difference(a: pd.Series, b: pd.Series):
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


def fix_order(cols: list):
    fish1 = cols[0]
    fish10 = cols[1]
    cols[0] = cols[-1]
    cols[1] = fish1
    cols[-1] = fish10
    return cols


def swap_columns(df: pd.DataFrame, cols_to_swap: dict):
    if cols_to_swap is None or len(cols_to_swap) == 0:
        return df
    cols = fix_order(df.columns.levels[0].tolist())
    for i in range(len(cols)):
        if cols_to_swap.get(cols[i]):
            df[cols[i]], df[cols_to_swap[cols[i]]] = df[cols_to_swap[cols[i]]], df[cols[i]]

    return df


def stitch_batches(batches_folder: str, debug=False):
    out = pd.DataFrame()
    batches = sorted(os.listdir(batches_folder))
    prev_dir = os.path.join(batches_folder, batches[0])
    csv_filepath = glob.glob(os.path.join(prev_dir, "*filtered.csv"))[0]
    prev_csv = pd.read_csv(csv_filepath, header=[0, 1, 2, 3])
    prev_csv.columns = prev_csv.columns.droplevel(0)
    prev_csv.reindex(columns=INDEX, fill_value=np.nan)

    for i in range(1, len(batches)):
        curr_dir = os.path.join(batches_folder, batches[i])
        try:
            curr_csv = pd.read_csv(glob.glob(os.path.join(curr_dir, "*filtered.csv"))[0], header=[0, 1, 2, 3])
            curr_csv.columns = curr_csv.columns.droplevel(0)
            curr_csv.reindex(columns=INDEX, fill_value=np.nan)
            curr_csv["individuals"] = curr_csv["individuals"] + prev_csv["individuals"].iloc[-1] + 1
            table = get_pairs(prev_csv.iloc[-1], curr_csv.iloc[0])
            if debug:
                print(f"prev: batch{i - 1}, curr: batch{i}. links: {table}")
            prev_csv = swap_columns(prev_csv, table)
            out = pd.concat((out, prev_csv), ignore_index=True)
            prev_csv = curr_csv
        except (FileNotFoundError, IndexError) as e:
            print(e)
            return

    out.to_csv(os.path.join(Path(batches_folder).parent.absolute(), Path(csv_filepath).name), index=False)

if __name__=="__main__":
    stitch_batches("/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc28_1_Tk3_022520/batches")