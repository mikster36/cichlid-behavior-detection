import numpy as np
import pandas as pd
import os

pd.options.mode.chained_assignment = None  # default='warn'


frames = ['/home/bree_student/Downloads/dlc_model-student-2023-07-26/labeled-data/'
          'MC_singlenuc23_1_Tk33_021220_0004_vid/CollectedData_student.csv']

"""
    Gets the average length of a cichlid from a trial based on labeled annotations

    Parameters
    frames: String paths to a .csv file of annotated data

    Returns
    A float representing the average length of a cichlid

"""

def euclidean_distance(row):
    x1, y1, x2, y2 = row[['nose-x', 'nose-y', 'backfin-x', 'backfin-y']]
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


"""
    Gets the average length of a cichlid from a trial based on labeled annotations
    
    Parameters
    frames: String paths to a .csv file of annotated data
    
    Returns
    A float representing the average length of a cichlid
    
"""


def avg_cichlid_length(frame):

    labels = pd.read_csv(frame, header=1)
    # get nose and backfin coordinates only
    labels = labels.loc[:, (labels.iloc[0] == 'nose') | (labels.iloc[0] == 'backfin')]
    # drop all empty coordinates
    all_nan = list(labels.iloc[2:].columns[labels.iloc[2:].isna().all()])
    labels = labels.drop(columns=all_nan)
    # rename columns to fish1, ..., fish10
    labels.columns = [col.split('.')[0] for col in labels.columns]
    # rename row 1's columns to nose-x, ..., backfin-y
    for i in range(len(labels.iloc[0])):
        labels.iloc[0, i] += f"-{labels.iloc[1, i]}"
    # drop x and y row
    labels = labels.drop(1)
    # rename columns to fish1-nose-x, ..., fish10-backfin-y
    labels.columns = [f"{labels.columns[i]}-{labels.iloc[0, i]}" for i in range(len(labels.columns))]
    # drop nose-x, ..., backfin-y row
    labels = labels.drop(0)

    # Get the unique fish numbers by splitting column names
    fish_columns = labels.columns.str.split('-').str[0].unique()

    # Create a new DataFrame to store reshaped data
    long_fish = pd.DataFrame()

    # Iterate through fish numbers and reshape data
    for fish_number in fish_columns:
        nose_x = f'{fish_number}-nose-x'
        nose_y = f'{fish_number}-nose-y'
        backfin_x = f'{fish_number}-backfin-x'
        backfin_y = f'{fish_number}-backfin-y'

        fish_data = {
            'nose-x': labels[nose_x],
            'nose-y': labels[nose_y],
            'backfin-x': labels[backfin_x],
            'backfin-y': labels[backfin_y]
        }

        fish_df = pd.DataFrame(fish_data)
        long_fish = pd.concat([long_fish, fish_df], ignore_index=True)

    long_fish = long_fish.dropna(how='any').astype(float)

    long_fish['dist'] = long_fish.apply(euclidean_distance, axis=1)

    mean = long_fish['dist'].mean()

    print(f'Mean: {mean}')
