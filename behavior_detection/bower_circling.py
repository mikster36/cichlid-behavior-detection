#!/usr/bin/env python
"""
    This script extracts potential 'bower circling' clips from a video
    @author: mikster36
    @date: 10/2/23
"""
import math
import os
from dataclasses import dataclass
from typing import Any, Union
import subprocess as s

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

import pandas as pd
import numpy as np
from numpy import ndarray, dtype, generic

matplotlib.use("TKAgg")


@dataclass
class Vel:
    direction: np.ndarray
    magnitude: float


@dataclass
class Fish:
    id: str
    position: Any
    vel: list[Vel]


@dataclass
class Track:
    a: Fish
    b: Fish
    start: str
    length: int


def video_to_frames(video: str):
    vid = cv2.VideoCapture(video)
    success, image = vid.read()
    count = 0
    output = os.path.join(os.path.dirname(video), "frames")
    if not os.path.exists(output):
        os.mkdir(output)
    if len(os.listdir(output)) > 0:
        return output
    while success:
        cv2.imwrite(f"{os.path.join(output, f'frame{count}.png')}", image)
        success, image = vid.read()
        count += 1
    return output


def show_nframes(frames: str, n: int):
    for i in range(n):
        image = cv2.imread(f"{os.path.join(frames, f'frame{i}.png')}")
        window_name = f'frame{i}'
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()


def point_in_focus(x, y, mask_x, mask_y, width, height) -> bool:
    return mask_x <= x <= mask_x + width and mask_y <= y <= mask_y + height


def shift_from_edge(x, y, width, height, debug=False) -> tuple:
    x_out = x
    y_out = y
    if x >= width - 80:
        x_out -= 80
    if y <= 10:
        y_out += 10
    if y >= height - 80:
        y_out -= 80
    if debug and (x != x_out or y != y_out):
        print(f"Old x, y: {(x, y)}. New x, y: {(x_out, y_out)}")
    return x_out, y_out


def fish_in_focus(fish, mask_xy: tuple, dimensions: tuple) -> bool:
    if dimensions is None:
        return True
    for cluster in fish:
        if not point_in_focus(cluster[0], cluster[1], mask_xy[0], mask_xy[1], dimensions[0], dimensions[1]):
            return False
    return True


def rotate(dx, dy, angle) -> tuple[float, float]:
    rad = math.radians(angle)
    ndx = dx * math.cos(rad) + dy * math.sin(rad)
    ndy = -dx * math.sin(rad) + dy * math.cos(rad)
    return ndx, ndy


def check_direction(vel: Vel, other: list[Vel], debug=False):
    avg_vel: np.ndarray = np.array((1 / len(other)) * sum(item.direction for item in other))
    avg_vel = avg_vel / np.linalg.norm(avg_vel)
    l_bound = rotate(avg_vel[0], avg_vel[1], 270)
    r_bound = rotate(avg_vel[0], avg_vel[1], 90)
    a_cross_b = l_bound[0] * vel.direction[1] - l_bound[1] * vel.direction[0]
    a_cross_c = l_bound[0] * r_bound[1] - l_bound[1] * r_bound[0]
    c_cross_b = r_bound[0] * vel.direction[1] - r_bound[1] * vel.direction[0]
    c_cross_a = r_bound[0] * l_bound[1] - r_bound[1] * l_bound[0]
    if a_cross_b * a_cross_c >= 0 and c_cross_b * c_cross_a >= 0:
        if debug: print(f"Altered direction.\nPrevious direction: {vel.direction}\nNew direction: {avg_vel}")
        vel.direction = avg_vel
    return vel


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

        Coordinate system:
                     img.width
             (0, 0) - - - - - - >
                   .
        img.height .   (x, y)**
                   .
                   v
        ** when plotting, this becomes (x, img.height - y) to conform to matplotlib's coordinate system
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
        xy = (0, 0)
    if show_mask:
        ax.add_patch(patches.Rectangle(xy=(xy[0], img.height - xy[1]), width=dimensions[0], height=-dimensions[1],
                                       alpha=0.2, fill=True, color="white"))

    for key, fish in fishes.items():
        fishcolor = color.get(key)
        if point_in_focus(x=fish.position[0][0], y=img.height - fish.position[0][1],
                          mask_x=xy[0], mask_y=xy[1], width=dimensions[0], height=dimensions[1]):
            text_xy = shift_from_edge(x=fish.position[0][0] + 8, y=img.height - fish.position[0][1],
                                      width=img.width, height=img.height, debug=True)
            ax.text(x=text_xy[0], y=text_xy[1], s=key, color='white',
                    fontsize='xx-small')
        # plot each body part's velocity
        for velocity, position in zip(fish.vel, fish.position):
            x, y = position
            # large change in position is likely not a correct track
            if velocity.magnitude >= img.width / 6 or velocity.magnitude > img.height / 6:
                break
            velocity.magnitude = 10 if velocity.magnitude < 10 else velocity.magnitude
            dx, dy = velocity.magnitude * velocity.direction
            dy = -dy
            ax.add_patch(patches.Arrow(x, img.height - y, dx=dx, dy=dy, width=5, color='white'))
            ax.plot(x, img.height - y, marker='.', color=fishcolor, markersize=1)

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


def get_centroid(xy_coords: list[tuple]) -> np.ndarray:
    x_sum, y_sum = 0, 0
    for i in xy_coords:
        if i[0] == 0 or i[0] is np.nan or i[1] == 0 or i[1] is np.nan:
            continue
        x_sum += i[0]
        y_sum += i[1]
    return np.array([x_sum / len(xy_coords), y_sum / len(xy_coords)])


def get_approximations(frame) -> dict[Any, ndarray[Any, dtype[Union[Union[generic, generic], Any]]]]:
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


def get_single_velocities(pi: np.ndarray, pf: np.ndarray, t: int) -> tuple[Vel, Vel, Vel]:
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


def df_to_reshaped_list(df: pd.DataFrame) -> list[dict[Fish: np.ndarray]]:
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


def same_fishes_in_t_frames(frames: list[dict[str: np.ndarray]], t) -> bool:
    fish_in_curr = set(frames[t - 1].keys())
    fish_in_prev = set(frames[0].keys())
    for i in range(1, len(frames)):
        fish_in_prev = fish_in_prev & set(frames[i].keys())
    return bool(fish_in_curr & fish_in_prev)


def _create_velocity_video(frames: str, fps=29):
    wd = os.getcwd()
    os.chdir(frames)
    args = ['ffmpeg', '-framerate', str(fps), '-pattern_type', 'glob', '-i',
            os.path.join(frames, "*.png"), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            os.path.join(frames, "velocity.mp4")]
    try:
        s.call(args=args, cwd=wd)
        print("Video successfully created.")
    except Exception as e:
        print(f"Video could not be successfully created.\nError: {e}")


def get_velocities(tracklets_path: str, smooth_factor=1, start_index=0, nframes=None,
                   mask_xy=(0, 0), mask_dimensions=None, save_as_csv=False) -> dict[str: dict]:
    tracklets: pd.DataFrame = pd.DataFrame(pd.read_hdf(tracklets_path))
    frames = df_to_reshaped_list(tracklets)
    nframes = nframes if nframes is not None else tracklets.shape[0] - start_index
    nwidth = int(math.log10(nframes)) + 1

    frames = [frames[i + start_index] for i in range(nframes)]
    allframes = {}
    t = smooth_factor + 1

    for i in range(t - 1, nframes, t - 1):
        t_frames = frames[i - t + 1:i + 1]
        bodies = [get_approximations(frame) for frame in t_frames]
        if not same_fishes_in_t_frames(bodies, t):
            continue
        pi, pf = bodies[0], bodies[t - 1]
        # gets the velocities of fish appearing in all t frames
        avg_vels = {key: get_single_velocities(pi.get(key), pf.get(key), t - 1)
                    for key in pf.keys() if key in pi}
        prev_frames = []
        for body in bodies[:t - 1]:  # can't plot velocity for the last frame
            for fish, vels in avg_vels.items():
                check_direction(vels[-1], vels[:-1], debug=True)
            prev_frames.append({fish: Fish(position=body.get(fish), vel=vel)
                                for fish, vel in avg_vels.items() if
                                fish_in_focus(body.get(fish), mask_xy, mask_dimensions)})
        for j, frame in enumerate(prev_frames):
            frame_index = i + j + start_index - (t - 1)
            # adjust frame number to be 0...0n instead of n
            frame_num = f"frame{frame_index :0{nwidth}d}"
            allframes.update({frame_num: frame})

    if len(allframes) > 0:
        print("Added velocities to tracklets.")
    return allframes


def create_velocity_video(video_path: str, tracklets_path: str, velocities=None, dest_folder=None, smooth_factor=1,
                          start_index=0, nframes=None, mask_xy=(0, 0), mask_dimensions=None, show_mask=False, fps=29,
                          save_as_csv=False, overwrite=False):
    # Only use this function after generating frames for the whole video. Otherwise, a shorter video will be made
    frames_path = video_to_frames(video_path)
    vel_path = dest_folder if dest_folder is not None else os.path.join(os.path.dirname(tracklets_path), "velocities")
    if not os.path.exists(vel_path):
        os.mkdir(vel_path)
    frames = velocities if velocities is not None else get_velocities(tracklets_path, smooth_factor, start_index,
                                                                      nframes, mask_xy, mask_dimensions, save_as_csv)
    i = 0
    vel_directory = os.listdir(vel_path)
    if len([frame for frame in vel_directory if frame.endswith(".png")]) == len(frames) and not overwrite:
        print("Velocities already plotted. Exiting...")
        return

    for frame_num, fishes in frames.items():
        frame_path = os.path.join(frames_path, f"frame{i}.png")
        i += 1
        try:
            plot_velocities(frame=frame_path, frame_num=frame_num, fishes=fishes, destfolder=vel_path, show=False,
                            xy=mask_xy, dimensions=mask_dimensions, show_mask=show_mask)
            print(f"Successfully plotted velocities for {frame_num}.")
        except Exception as e:
            print(f"Could not plot velocities for {frame_num}.\nError: {e}")


def track_bower_circling(frames: dict[str: dict[str: Fish]]):
    tracks = {}
    for frame_num, frame in frames.items():
        fish_nums = list(frame.keys())
        fishes = list(frame.values())
        matched = set()
        # check every combination of fish and exit when one pair is made (BC cannot happen between more than two fish)
        # this may be changed later if correct pairs are being missed
        for i in range(len(fishes) - 1):
            if fish_nums[i] in matched:
                continue
            a = fishes[i]

            for j in range(i + 1, len(fishes)):
                if fish_nums[j] in matched:  # this fish already has a pair, so skip
                    continue

                b = fishes[j]
                distance = np.linalg.norm(a.position - b.position)
                if distance > 100:  # fish must be within 100 px of one another
                    continue

                ahead_btail = np.linalg.norm(a.position[0] - b.position[-1])
                atail_bhead = np.linalg.norm(a.position[-1] - b.position[0])
                # a's head must be close to b's tail and a's tail must be close to b's head
                if ahead_btail > 50 or atail_bhead > 50:
                    continue
                if (fish_nums[i] not in matched or  # fish doesn't have a match (should be ensured by previous checks)
                        distance < np.linalg.norm(tracks.get(a).a.position - tracks.get(a).b.position)):
                    if tracks.get(fish_nums[i]):
                        tracks[fish_nums[i]].length += 1
                    else:
                        tracks.update({fish_nums[i]: Track(a=a, b=b, start=frame_num, length=1)})
                    matched.add(fish_nums[i])
                    matched.add(fish_nums[j])
                    break
    print(tracks)


if __name__ == "__main__":
    frames = {"frame001": {
        "fish1": Fish(id="fish1", position=np.array([[1, 2], [2, 3], [4, 5]]), vel=None),
        "fish2": Fish(id="fish2", position=np.array([[600, 600], [500, 500], [400, 400]]), vel=None),
        "fish5": Fish(id="fish5", position=np.array([[13, 10], [14, 14], [16, 17]]), vel=None),
        "fish3": Fish(id="fish3", position=np.array([[10, 10], [11, 11], [12, 12]]), vel=None),
        "fish4": Fish(id="fish4", position=np.array([[11, 9], [15, 11], [12, 16]]), vel=None)
    }, "frame002": {
        "fish1": Fish(id="fish1", position=np.array([[1, 3], [2, 4], [4, 6]]), vel=None),
        "fish2": Fish(id="fish2", position=np.array([[650, 600], [550, 500], [450, 400]]), vel=None),
        "fish5": Fish(id="fish5", position=np.array([[14, 10], [15, 14], [17, 17]]), vel=None),
        "fish3": Fish(id="fish3", position=np.array([[14, 12], [15, 13], [11, 10]]), vel=None),
        "fish4": Fish(id="fish4", position=np.array([[9, 8], [12, 11], [15, 14]]), vel=None)
    }}
    track_bower_circling(frames)
