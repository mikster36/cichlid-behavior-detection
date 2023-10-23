from pathlib import Path

import deeplabcut as dlc
import pandas as pd
from os import path


def analyze_video(config_path, video_path, n_fish=None):
    video_path = str(video_path)
    dlc.analyze_videos(config_path, [video_path], auto_track=False, robust_nframes=True)
    dlc.convert_detections2tracklets(config_path, [video_path], track_method='ellipse')
    if n_fish is None:
        n_fish = 3
        while n_fish > 0:
            try:
                print(f'attempting stitching with n_tracks={n_fish}')
                dlc.stitch_tracklets(config_path, [video_path], n_tracks=n_fish)
                break
            except ValueError as e:
                print(f'failed to stitch tracklets with n_fish={n_fish}')
                print(e)
                n_fish -= 1
        if n_fish == 0:
            print('stitching failed')
            return
    else:
        dlc.stitch_tracklets(config_path, [video_path], n_tracks=n_fish)
    dlc.filterpredictions(config_path, video_path)
    print(f'analyzed {Path(video_path).name} successfully')


def fix_individual_names(video_path):
    h5_path = str(next(Path(video_path).parent.glob('*_filtered.h5')))
    csv_path = h5_path.replace('.h5', '.csv')
    df = pd.read_hdf(h5_path)
    df.rename(columns={'ind1': 'individual1', 'ind2': 'individual2', 'ind3': 'individual3'}, inplace=True)
    df.to_csv(csv_path)
    df.to_hdf(h5_path, "df_with_missing", format="table", mode="w")