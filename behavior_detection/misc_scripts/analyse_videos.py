from pathlib import Path
#from behavior_detection.misc_scripts.train_network import kill_and_reset
import behavior_detection.misc_scripts.ffmpeg_split as ffmpeg_split

import pandas as pd
import deeplabcut as dlc
import subprocess

import random
import os


def get_subfolders(folder):
    ''' returns list of subfolders '''
    return [os.path.join(folder, p) for p in os.listdir(folder) if os.path.isdir(os.path.join(folder, p))]


"""
Splits each video in path into 1 hour batches

str path: path to directory with videos
bool long: set True if videos are longer than an hour
bool exact: set True if you want the chunks to be exactly 1 hour long
            set to False for faster but less exact chunks
"""


def split_videos_by_hour(path=str(), long=True, exact=False):
    subfolders = get_subfolders(path)
    basedir = path
    for subfolder in subfolders:
        subfoldername = subfolder[subfolder.rindex('/') + 1:]
        if not subfoldername.startswith('MC'):
            continue
        batches = os.path.join(subfolder, 'batches')
        if not os.path.exists(batches):
            os.mkdir(batches)
        if long:
            vcodec = 'copy' if not exact else 'h264'
            for file in os.listdir(subfolder):
                if file.endswith('_vid.mp4'):
                    ffmpeg_split.split_by_seconds(subfolder + f'/{file}', 3600, vcodec=vcodec)


def format_time(time: int):
    hh = str(int(time / 3600) % 60)
    hh = "0" + hh if len(hh) == 1 else hh
    mm = str(int(time / 60) % 60)
    mm = "0" + mm if len(mm) == 1 else mm
    ss = str(time % 60)
    ss = "0" + ss if len(ss) == 1 else ss
    return f'{hh}:{mm}:{ss}'


def generate_random_clips(videos=[], clip_length=10, n=10):
    if videos is None:
        raise ValueError("No videos in list.")
    if type(videos) is not list:
        raise ValueError("Videos must be in a list.")
    if n == 0:
        print("No clips created.")
        return
    for video in videos:
        temp_dir = str(Path(video).parent)
        if not os.path.exists(os.path.join(temp_dir, "../testing")):
            os.mkdir(os.path.join(temp_dir, "../testing"))
        vid_length = ffmpeg_split.get_video_length(video)
        temp_dir = os.path.join(temp_dir, "../testing")
        for i in range(n):
            if os.path.exists(os.path.join(temp_dir, f"testing{i}")):
                print(f'test{i} already exits. Skipping...')
                continue
            os.mkdir(os.path.join(temp_dir, f"testing{i}"))
            temp_dir = os.path.join(temp_dir, f"testing{i}")
            start_time = int(random.random() * (vid_length - clip_length))
            args = ['ffmpeg',
                    '-ss', format_time(start_time),
                    '-i', video, '-t',
                    format_time(clip_length),
                    '-c:v', 'copy', '-c:a', 'copy', str(os.path.join(temp_dir, f'test{i}.mp4'))]
            subprocess.call(args)
            print(f'Clip {i} successfully made.')
            temp_dir = Path(temp_dir).parent  # move to uproot


def analyze_video(config_path, video_path):
    video_path = str(video_path)
    dlc.analyze_videos(config_path, [video_path], allow_growth=True, auto_track=False, robust_nframes=True, shuffle=4, save_as_csv=True)
    dlc.convert_detections2tracklets(config_path, [video_path], track_method='ellipse', shuffle=4)
    n_fish = 10
    while n_fish > 0:
        try:
            print(f'attempting stitching with n_tracks={n_fish}')
            dlc.stitch_tracklets(config_path, [video_path], n_tracks=n_fish, shuffle=4, save_as_csv=True)
            break
        except (ValueError, IOError) as e:
            print(f'failed to stitch tracklets with n_fish={n_fish}')
            print(e)
            n_fish -= 1
    if n_fish == 0:
        print('stitching failed')
        dlc.create_video_with_all_detections(config_path, [video_path], shuffle=4)
        return 0
    fix_individual_names(video_path)
    dlc.filterpredictions(config_path, video_path, shuffle=4)
    print(f'analyzed {Path(video_path).name} successfully')
    return n_fish


def column_mapping(col=str()):
    if col.startswith('ind') and col[3:].isdigit():
        index = int(col[3:])
        if 1 <= index <= 10:
            return f'fish{index}'
    return col


def fix_individual_names(video_path):
    h5_path = str(next(Path(video_path).parent.glob('*_el.h5')))
    csv_path = h5_path.replace('.h5', '.csv')
    df = pd.DataFrame(pd.read_hdf(h5_path))
    df.rename(columns={col[1]: column_mapping(col[1]) for col in df.columns}, inplace=True)
    df.to_csv(csv_path)
    df.to_hdf(h5_path, "df_with_missing", format="table", mode="w")


def short_testing():
    config_path = '/home/bree_student/Downloads/dlc_model-student-2023-07-26/config.yaml'
    trial_folder = '/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520'
    clips = [os.path.join(trial_folder, f) for f in os.listdir(trial_folder) if f.endswith('.mp4')]
    generate_random_clips(clips)
    for f in os.scandir(trial_folder):
        if f.is_dir() & f.path.endswith("testing"):
            for vid_folder in os.scandir(f):
                vid = os.path.join(vid_folder.path, f"{os.path.basename(vid_folder).replace('ing', '')}.mp4")
                n_fish = analyze_video(config_path, vid)
                #kill_and_reset()
                displayedinidividuals = [f'fish{i}' for i in range(1, n_fish + 1)]
                dlc.plot_trajectories(config_path, [vid], shuffle=4,
                                      displayedindividuals=displayedinidividuals)
                dlc.create_labeled_video(config_path, [vid], shuffle=4, filtered=True,
                                         displayedindividuals=displayedinidividuals, color_by="individual")


def analyse_videos(config_path, vid):
    n_fish = analyze_video(config_path, vid)
    #kill_and_reset()
    displayedinidividuals = [f'fish{i}' for i in range(1, n_fish + 1)]
    dlc.plot_trajectories(config_path, [vid], shuffle=4,
                          displayedindividuals=displayedinidividuals)
    dlc.create_labeled_video(config_path, [vid], shuffle=4, filtered=True,
                             displayedindividuals=displayedinidividuals, color_by="individual")