#!/usr/bin/env python
"""
    This script extracts potential 'bower circling' clips from a video
    @author: mikster36
    @date: 10/2/23
"""
import itertools
import os
from dataclasses import dataclass
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.markers as markers
from PIL import Image

import pandas as pd
import numpy as np

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
    position: list[np.ndarray]
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


def show_nframes(frames: str, n: int):
    import cv2
    for i in range(n):
        image = cv2.imread(f"{os.path.join(frames, f'frame{i}.png')}")
        window_name = f'frame{i}'
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()


def plot_velocities(frame: str, fishes: list[Fish], destfolder: str, show=False):
    img = Image.open(frame)
    img_data = np.flipud(np.array(img))
    fig, ax = plt.subplots()
    ax.imshow(img_data, origin='upper')
    color = itertools.cycle(('red', 'blue'))

    for i, fish in enumerate(fishes):
        # plot each body part's velocity
        fishcolor = next(color)
        for velocity, position in zip(fish.vel, fish.position):
            x, y = position
            y = img.height - y
            dx, dy = velocity.magnitude * velocity.direction
            dy = -dy
            ax.add_patch(patches.Arrow(x, y, dx=dx, dy=dy, width=5, color='white'))
            ax.plot(x, y, marker='.', color=fishcolor, markersize=1)
        print()

    ax.set_xlim(0, img.width)
    ax.set_ylim(0, img.height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(os.path.join(destfolder, f"{os.path.basename(frame).split('.')[0]}-velocities.png"),
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    if show:
        plt.imshow()


def get_centroid(xy_coords: list[tuple]):
    x_sum, y_sum = 0, 0
    for i in xy_coords:
        x_sum += i[0]
        y_sum += i[1]
    return np.array([x_sum / len(xy_coords), y_sum / len(xy_coords)])


def get_approximations(frame):
    bodies = []
    for individual in frame:
        # front cluster is the centroid of nose, lefteye, righteye, and spine1
        # middle cluster is the centroid of spine2, spine3, leftfin, and rightfin
        # tail is the backfin
        # these approximations help abstract the fish and its movement
        front_cluster = [(individual[i][0], individual[i][1]) for i in range(4)]
        middle_cluster = [(individual[i][0], individual[i][1]) for i in range(3, 9) if i != 6]
        front = get_centroid(front_cluster)
        centre = get_centroid(middle_cluster)
        tail = (individual[6][0], individual[6][1])
        bodies.append(np.array([front, centre, tail]))
    return bodies


def get_velocities(pi: np.ndarray, pf: np.ndarray, n: int):
    """
    Gets the velocity of each fish in a frame

    Args:
        pi (tuple [3-tuple of tuples]): the initial position of a cichlid (head, centre, tail)
        pf (tuple [3-tuple of tuples]): the final position of a cichlid (head, centre, tail)
        n (int): the time (in frames) between final and initial position
    Returns:
        tuple (3-tuple of tuples): the velocity for each body mass in the form: (direction, magnitude)
    """
    front_vel = np.array([pf[0][0] - pi[0][0], pf[0][1] - pi[0][1]]) / n
    centre_vel = np.array([pf[1][0] - pi[1][0], pf[1][1] - pi[1][1]]) / n
    tail_vel = np.array([pf[2][0] - pi[2][0], pf[2][1] - pi[2][1]]) / n
    return (Vel(front_vel / np.linalg.norm(front_vel), np.linalg.norm(front_vel)),
            Vel(centre_vel / np.linalg.norm(centre_vel), np.linalg.norm(centre_vel)),
            Vel(tail_vel / np.linalg.norm(tail_vel), np.linalg.norm(tail_vel)))


if __name__=="__main__":
    data_pickle = pd.read_pickle(filepath_pickle)
    start_index = 70
    nframes = 200
    frames = [data_pickle[i + start_index] for i in range(nframes)]

    for i in range(1, 60):
        prev_frame, curr_frame = frames[i - 1], frames[i]
        prev_bodies, curr_bodies = get_approximations(prev_frame), get_approximations(curr_frame)
        if len(prev_bodies) != len(curr_bodies):
            continue
        vels = [get_velocities(prev_bodies[i], curr_bodies[i], 1) for i in range(len(curr_bodies))]
        fishes = [Fish(position=i, vel=j) for i, j in zip(prev_bodies, vels)]
        frame_path = (f"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520"
                      f"/bower_circling/frames/")
        frame_path = os.path.join(frame_path, f"frame{i + start_index - 1}.png")
        dest_folder = (f"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520"
                       f"/bower_circling/velocities/")

        plot_velocities(frame=frame_path, fishes=fishes, destfolder=dest_folder, show=False)
