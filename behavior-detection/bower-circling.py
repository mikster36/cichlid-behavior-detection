#!/usr/bin/env python
"""
    This script extracts potential 'bower circling' clips from a video
    @author: mikster36
    @date: 10/2/23
"""
import itertools
import math
import os
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import pandas as pd
import numpy as np

filepath_h5 = (r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/"
               r"bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_el_filtered.h5")
filepath_pickle = (r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520"
                   r"/bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_assemblies.pickle")
video = (r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520"
         r"/bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_el_filtered_id_labeled.mp4")
out = (r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/bower_circling"
       r"/labeled-frames")
matplotlib.use("TKAgg")


@dataclass
class Vel:
    direction: np.ndarray
    magnitude: float


@dataclass
class Fish:
    position: Any
    vel: list[Vel]


def read_video(video: str, output: str):
    import cv2
    vid = cv2.VideoCapture(video)
    success, image = vid.read()
    count = 0
    while success:
        cv2.imwrite(f"{os.path.join(output, f'frame{count}.png')}", image)
        success, image = vid.read()
        print('Read a new frame: ', success)
        count += 1


def create_velocity_video(frames: str, framerate=30):
    # Only use this function after generating frames for the whole video. Otherwise, a shorter video will be made
    import subprocess as s
    wd = os.getcwd()
    os.chdir(frames)
    args = ['ffmpeg', '-framerate', str(framerate), '-pattern_type', 'glob', '-i',
            os.path.join(frames, "*.png"), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            os.path.join(frames, "velocity.mp4")]
    try:
        s.call(args=args, cwd=wd)
        print("Video successfully created.")
    except Exception as e:
        print(f"Video could not be successfully created.\nError: {e}")


def show_nframes(frames: str, n: int):
    import cv2
    for i in range(n):
        image = cv2.imread(f"{os.path.join(frames, f'frame{i}.png')}")
        window_name = f'frame{i}'
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()


def in_focus(x, y, mask_x, mask_y, width, height):
    return mask_x <= x <= mask_x + width and mask_y <= y <= mask_y + height


def plot_velocities(frame: str, frame_num: str, fishes: dict[str: Fish], destfolder: str, show=False,
                    xy=None, dimensions=None, show_mask=False):
    """
        Plots the velocities of each fish in a given area of focus. Since there are likely fish outside
        the area where a behaviour can occur, the user can specify which part of the frame can be ignored

        Args:
            frame: str - the path of the frame
            frame_num: str - the frame number with respect to all frames e.g. 0043 is the frame number for
            a trial with 1342 frames
            fishes: dict["fishn": Fish object] - a dictionary of all the fish in a frame
            destfolder: str - the path of the folder where the plotted velocities will be stored
            show: bool - whether to display the velocities plot
            xy: tuple - (x, y) coordinates of the bottom left of the rectangle mask (area in focus)
            dimensions: tuple - (width, height) of the rectangle mask (area in focus)
            show_mask: bool - whether to display the mask on the produced image
    """
    img = Image.open(frame)
    img_data = np.flipud(np.array(img))
    fig, ax = plt.subplots()
    ax.imshow(img_data, origin='upper')
    color = {'fish1': 'red', 'fish2': 'orange', 'fish3': 'yellow', 'fish4': 'green', 'fish5': 'blue',
             'fish6': 'purple', 'fish7': 'pink', 'fish8': 'brown', 'fish9': 'white', 'fish10': 'black'}
    if dimensions is None:
        dimensions = (img.width, img.height)
    if xy is None:
        xy = (img.width, img.height)
    if show_mask:
        ax.add_patch(patches.Rectangle(xy=(199, img.height - 240), width=dimensions[0], height=-dimensions[1],
                                       alpha=0.2, fill=True, color="white"))

    for key, fish in fishes.items():
        fishcolor = color.get(key)
        if in_focus(x=fish.position[0][0], y=img.height - fish.position[0][1],
                    mask_x=xy[0], mask_y=xy[1], width=dimensions[0], height=dimensions[1]):
            ax.text(x=fish.position[0][0] + 8, y=img.height - fish.position[0][1], s=key, color='white',
                    fontsize='xx-small')
        # plot each body part's velocity
        for velocity, position in zip(fish.vel, fish.position):
            x, y = position
            y = img.height - y
            if not in_focus(x, y, xy[0], img.height - xy[1], dimensions[0], dimensions[1]):
                continue
            # large change in position is likely not a correct track
            elif velocity.magnitude > img.width / 4 or velocity.magnitude > img.height / 4:
                break
            dx, dy = velocity.magnitude * velocity.direction
            dy = -dy
            ax.add_patch(patches.Arrow(x, y, dx=dx, dy=dy, width=5, color='white'))
            ax.plot(x, y, marker='.', color=fishcolor, markersize=1)

    ax.plot()
    ax.set_xlim(0, img.width)
    ax.set_ylim(0, img.height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(os.path.join(destfolder, f"{frame_num}-velocities.png"),
                dpi=300, bbox_inches='tight', pad_inches=0)
    if show:
        plt.imshow()
    plt.close()


def get_centroid(xy_coords: list[tuple]):
    x_sum, y_sum = 0, 0
    for i in xy_coords:
        if i[0] == 0 or i[0] is np.nan or i[1] == 0 or i[1] is np.nan:
            continue
        x_sum += i[0]
        y_sum += i[1]
    return np.array([x_sum / len(xy_coords), y_sum / len(xy_coords)])


def get_approximations(frame):
    """
        Approximates each fish in a frame to three clusters: front, middle, tail

        Args:
            frame: list of dicts - a frame with each detection
        Returns:
            a list of dicts where each key is a fish and its value is a matrix of its
            body clusters' positions and likelihoods
    """
    bodies = {}
    for fish, matrix in frame.items():
        # front cluster is the centroid of nose, lefteye, righteye, and spine1
        # middle cluster is the centroid of spine2, spine3, leftfin, and rightfin
        # tail is the backfin
        # these approximations help abstract the fish and its movement
        front_cluster = [(matrix[i][0], matrix[i][1]) for i in range(4)]
        middle_cluster = [(matrix[i][0], matrix[i][1]) for i in range(3, 9) if i != 6]
        front = get_centroid(front_cluster)
        centre = get_centroid(middle_cluster)
        tail = (matrix[6][0], matrix[6][1])
        bodies.update({fish: np.array([front, centre, tail])})
    return bodies


def get_velocities(pi: np.ndarray, pf: np.ndarray, t: int):
    """
    Gets the velocity of a fish in a frame by calculating the velocity of each body mass

    Args:
        pi (tuple [3-tuple of tuples]): the initial position of a cichlid (head, centre, tail)
        pf (tuple [3-tuple of tuples]): the final position of a cichlid (head, centre, tail)
        t (int): the time (in frames) between final and initial position
    Returns:
        tuple (3-tuple of tuples): the velocity for each body mass in the form: (direction, magnitude)
    """
    front_vel = np.array([pf[0][0] - pi[0][0], pf[0][1] - pi[0][1]]) / t
    centre_vel = np.array([pf[1][0] - pi[1][0], pf[1][1] - pi[1][1]]) / t
    tail_vel = np.array([pf[2][0] - pi[2][0], pf[2][1] - pi[2][1]]) / t
    return (Vel(front_vel / np.linalg.norm(front_vel), np.linalg.norm(front_vel)),
            Vel(centre_vel / np.linalg.norm(centre_vel), np.linalg.norm(centre_vel)),
            Vel(tail_vel / np.linalg.norm(tail_vel), np.linalg.norm(tail_vel)))


def df_to_reshaped_list(df: pd.DataFrame):
    """
    By default, the *_el.h5 file is stored as a DataFrame with a shape and organisation
    similar to how the csv file looks, i.e.

    individual  fish1 fish1 fish1       ...
    bodypart    nose  nose  nose        ...
    coords      x     y     likelihood  ...
    frame #
    ...

    This method reshapes that data into a list of dicts, where the key is the fish and
    its value is a matrix of the following shape. The frame number is the index of the dict

                x   y   likelihood
    nose
    lefteye
    ...
    rightfin
    """
    frames = list()
    for i in range(df.shape[0]):
        frame = df.iloc[i].droplevel(0).dropna()
        frame = frame.unstack(level=[0, 2])
        frame: pd.DataFrame = frame.reindex(['nose', 'lefteye', 'righteye',
                                             'spine1', 'spine2', 'spine3',
                                             'backfin', 'leftfin', 'rightfin'])
        framedict = {fish: frame[fish].to_numpy() for fish in frame.columns.get_level_values(0).unique()}
        frames.append(framedict)
    return frames


def same_fishes_in_t_frames(frames: list[dict[str: np.ndarray]], t):
    fish_in_curr = set(frames[t - 1].keys())
    fish_in_prev = set(frames[0].keys())
    for i in range(1, len(frames)):
        fish_in_prev = fish_in_prev & set(frames[i].keys())
    return bool(fish_in_curr & fish_in_prev)


if __name__ == "__main__":
    frames_path = (r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/"
                   r"bower_circling/frames")
    dest_folder = (r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520"
                   r"/bower_circling/velocities/")
    tracklets: pd.DataFrame = pd.DataFrame(pd.read_hdf(filepath_h5))
    frames = df_to_reshaped_list(tracklets)
    start_index = 0
    nframes = tracklets.shape[0] - start_index
    width = int(math.log10(nframes)) + 1
    smooth_velocity = True
    t = 6 if smooth_velocity else 2

    frames = [frames[i + start_index] for i in range(nframes)]

    for i in range(t - 1, nframes, t - 1):
        t_frames = frames[i - t + 1:i + 1]
        bodies = [get_approximations(frame) for frame in t_frames]
        if not same_fishes_in_t_frames(bodies, t):
            continue
        pi, pf = bodies[0], bodies[t - 1]
        # gets the velocities of fish appearing in all t frames
        avg_vels = {key: get_velocities(pi.get(key), pf.get(key), t)
                    for key in pf.keys() if key in pi}
        prev_frames = []
        for body in bodies[:t - 1]:  # can't plot velocity for the last frame
            prev_frames.append({fish: Fish(position=body.get(fish), vel=vel) for fish, vel in avg_vels.items()})
        for j, frame in enumerate(prev_frames):
            frame_index = i + j + start_index - (t - 1)
            frame_path = os.path.join(frames_path, f"frame{frame_index}.png")
            # adjust frame number to be 0...0n instead of n
            frame_num = f"{(frame_index):0{width}d}"
            try:
                plot_velocities(frame=frame_path, frame_num=frame_num, fishes=frame, destfolder=dest_folder,
                                show=False, xy=(197, 1071), dimensions=(938, 698))
                print(f"Successfully plotted velocities for frame {frame_index}.")
            except Exception as e:
                print(f"Could not plot velocities for frame {frame_index}.\nError: {e}")
    create_velocity_video(frames=dest_folder)
