#!/usr/bin/env python
"""
    This script extracts potential 'bower circling' clips from a video
    @author: mikster36
    @date: 10/2/23
"""
import math
import os
import sys
import typing
from dataclasses import dataclass
import subprocess as s
from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

from tqdm import tqdm
import pandas as pd
import numpy as np
from behavior_detection.misc_scripts.ffmpeg_split import get_video_length

matplotlib.use("TKAgg")
np.seterr(divide='ignore', invalid='ignore')


@dataclass
class Vel:
    direction: np.ndarray
    magnitude: float


@dataclass
class Fish:
    id: str
    position: typing.Any
    vel: typing.List[Vel]
    bc: bool


@dataclass
class Track:
    """
        This is a class to store data for a potential bower circling track

        Attributes:
            a: Fish
                the fish object representing the first fish in the pair
            b: Fish
                the fish object representing the second fish in the pair
            start: str
                which frame the track began on
            end: str
                which frame the track ends on / was last updated on
            length: int
                the number of consecutive frames for which the track has existed
    """
    a: Fish
    b: Fish
    start: str
    end: str
    length: int

    def is_dead(self, frame: str, t: int):
        curr_frame = str_to_int(frame)
        prev_frame = str_to_int(self.end)
        return curr_frame - prev_frame > t


def str_to_int(s: str):
    out = str()
    for c in s:
        if c.isdigit():
            out += c
    return int(out)


def video_to_frames(video: str):
    print("Getting frames from video...")
    vid = cv2.VideoCapture(video)
    success, image = vid.read()
    total_frames = int(get_video_fps(video) * get_video_length(video))
    width = int(math.log10(total_frames)) + 1
    output = os.path.join(os.path.dirname(video), "frames")
    count = 0
    if not os.path.exists(output):
        os.mkdir(output)
    if len(os.listdir(output)) > 0:
        return output
    while success:
        cv2.imwrite(os.path.join(output, f'frame{count:0{width}d}.png'), image)
        success, image = vid.read()
        count += 1
    return output


def get_video_fps(video: str):
    vid = cv2.VideoCapture(video)
    return vid.get(cv2.CAP_PROP_FPS)


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
    if fish is None:
        return False
    if dimensions is None:
        return True
    for cluster in fish:
        if cluster is None:
            continue
        if not point_in_focus(cluster[0], cluster[1], mask_xy[0], mask_xy[1], dimensions[0], dimensions[1]):
            return False
    return True


def rotate(dx, dy, angle) -> typing.Tuple[float, float]:
    rad = math.radians(angle)
    ndx = dx * math.cos(rad) + dy * math.sin(rad)
    ndy = -dx * math.sin(rad) + dy * math.cos(rad)
    return ndx, ndy


def check_direction(vel: Vel, other: typing.List[Vel], debug=False):
    avg_vel: np.ndarray = np.array((1 / len(other)) * sum(item.direction for item in other))
    avg_vel = avg_vel / np.linalg.norm(avg_vel)
    if np.isnan(avg_vel).any():
        return vel
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


def plot_velocities(frame: str, frame_num: str, fishes: typing.Dict[typing.AnyStr, Fish], destfolder: str, show=False,
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
        """if fish.bc:
            ax.text(x=fish.position[0][0] - 8, y=img.height - fish.position[0][1] - 10,
                    s="Bower circling", color="white", fontsize="medium")
        """
        fishcolor = color.get(key)
        if point_in_focus(x=fish.position[0][0], y=img.height - fish.position[0][1],
                          mask_x=xy[0], mask_y=xy[1], width=dimensions[0], height=dimensions[1]):
            text_xy = shift_from_edge(x=fish.position[0][0] + 8, y=img.height - fish.position[0][1],
                                      width=img.width, height=img.height)
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

    # for further optimisation, adjust the video making process to not rely on images being saved for each frame.
    # rather, use a stream to create the video without saving to disk
    # ref: https://stackoverflow.com/questions/73609006/how-to-create-a-video-out-of-frames-without-saving-it-to-disk-using-python
    # ref: https://stackoverflow.com/questions/4092927/generating-movie-from-python-without-saving-individual-frames-to-files?rq=3
    plt.savefig(os.path.join(destfolder, f"{frame_num}.png"),
                dpi=300, bbox_inches='tight', pad_inches=0)
    if show:
        plt.imshow()
    plt.close()


def get_centroid(xy_coords: typing.List[typing.Tuple]) -> np.ndarray:
    if len(xy_coords) == 0:
        return None
    x_sum, y_sum = 0, 0
    for i in xy_coords:
        if i[0] is np.nan or i[1] is np.nan:
            continue
        x_sum += i[0]
        y_sum += i[1]

    x_out = x_sum / len(xy_coords)
    y_out = y_sum / len(xy_coords)
    if np.isnan(x_out) or np.isnan(y_out):
        return None
    return np.array([x_sum / len(xy_coords), y_sum / len(xy_coords)])


def get_single_velocities(pi: np.ndarray, pf: np.ndarray, t: int) -> typing.Tuple[Vel, Vel, Vel]:
    """
    Gets the velocity of a fish in a frame by calculating the velocity of each body mass

    Args:
        pi (tuple [3-tuple of tuples]): the initial position of a cichlid (head, centre, tail)
        pf (tuple [3-tuple of tuples]): the final position of a cichlid (head, centre, tail)
        t (int): the time (in frames) between final and initial position
    Returns:
        tuple (3-tuple of tuples): the velocity for each body mass in the form: (direction, magnitude)
    """
    try:
        front_vel = np.array([pf[0][0] - pi[0][0], pf[0][1] - pi[0][1]]) / t
    except TypeError:
        front_vel = 0
    try:
        centre_vel = np.array([pf[1][0] - pi[1][0], pf[1][1] - pi[1][1]]) / t
    except TypeError:
        centre_vel = 0
    tail_vel = np.array([pf[2][0] - pi[2][0], pf[2][1] - pi[2][1]]) / t
    return (Vel(front_vel / np.linalg.norm(front_vel), np.linalg.norm(front_vel)),
            Vel(centre_vel / np.linalg.norm(centre_vel), np.linalg.norm(centre_vel)),
            Vel(tail_vel / np.linalg.norm(tail_vel), np.linalg.norm(tail_vel)))


def none_count(a: list):
    c = 0
    for i in a:
        if i is None:
            c += 1
    return c


def get_approximations(series: pd.Series) -> typing.Dict[typing.AnyStr, typing.Any]:
    """
        Approximates each fish in a frame to three clusters: front, middle, tail

        Args:
            series: pd.Series
                A row from the h5 tracklets file
        Returns:
            a list of dicts where each key is a fish and its value is a matrix of its
            body clusters' positions and likelihoods
    """
    bodies = {}
    for i in range(0, series.shape[0], 27):
        # front cluster is the centroid of nose, lefteye, righteye, and spine1
        # middle cluster is the centroid of spine2, spine3, leftfin, and rightfin
        # tail is the backfin
        # these approximations help abstract the fish and its movement
        if series.iloc[i:i + 27].isna().sum() > 18:
            continue
        front_cluster = [(series.iloc[i + j], series.iloc[i + j + 1]) for j in range(0, 12, 3)]
        centre_cluster = [(series.iloc[i + j], series.iloc[i + j + 1]) for j in range(12, 27, 3) if j != 18]
        if np.isnan(series.iloc[i + 18]) or np.isnan(series.iloc[i + 19]):
            tail = None
        else:
            tail = (series.iloc[i + 18], series.iloc[i + 19])
        out = [get_centroid(front_cluster), get_centroid(centre_cluster), tail]
        if none_count(out) > 0:
            continue
        bodies.update({f"fish{int(i / 27 + 1)}": out})
    return bodies


def same_fishes_in_t_frames(frames: typing.List[typing.Dict[typing.AnyStr, np.ndarray]], t) -> bool:
    fish_in_curr = set(frames[t - 1].keys())
    fish_in_prev = set(frames[0].keys())
    for i in range(1, len(frames)):
        fish_in_prev = fish_in_prev & set(frames[i].keys())
    return bool(fish_in_curr & fish_in_prev)


def _create_velocity_video(frames: str):
    wd = os.getcwd()
    os.chdir(frames)
    args = ['ffmpeg', '-framerate', str(get_video_fps(frames)), '-pattern_type', 'glob', '-i',
            os.path.join(frames, "*.png"), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            os.path.join(frames, "velocity.mp4")]
    try:
        s.call(args=args, cwd=wd)
        print("Video successfully created.")
    except Exception as e:
        print(f"Video could not be successfully created.\nError: {e}")


def get_velocities(tracklets_path: str, smooth_factor=1, mask_xy=(0, 0), mask_dimensions=None,
                   save_as_csv=False) -> typing.Dict[typing.AnyStr, typing.Dict]:
    tracklets: pd.DataFrame = pd.DataFrame(pd.read_hdf(tracklets_path))
    tracklets.columns = tracklets.columns.droplevel(0)
    nwidth = int(math.log10(tracklets.shape[0])) + 1

    allframes = {}
    t = smooth_factor + 1

    for i in tqdm(range(t - 1, tracklets.shape[0], t - 1), desc="Getting velocities..."):
        t_frames = tracklets.iloc[[j for j in range(i - t + 1, i + 1)]]
        bodies = [get_approximations(row) for index, row in t_frames.iterrows()]
        if not same_fishes_in_t_frames(bodies, t):
            continue
        pi, pf = bodies[0], bodies[t - 1]
        # gets the velocities of fish appearing in all t frames
        avg_vels = {key: get_single_velocities(pi.get(key), pf.get(key), t - 1)
                    for key in pf.keys() if key in pi}
        prev_frames = []
        for body in bodies[:t - 1]:  # can't plot velocity for the last frame
            for fish, vels in avg_vels.items():
                check_direction(vels[-1], vels[:-1])
            prev_frames.append({fish: Fish(id=fish, position=body.get(fish), vel=vel, bc=False)
                                for fish, vel in avg_vels.items() if
                                fish_in_focus(body.get(fish), mask_xy, mask_dimensions)})
        for j, frame in enumerate(prev_frames):
            frame_index = i + j - (t - 1)
            # adjust frame number to be 0...0n instead of n
            frame_num = f"frame{frame_index :0{nwidth}d}"
            allframes.update({frame_num: frame})

    if len(allframes) > 0:
        print("Added velocities to tracklets.")
    else:
        print("Could not add velocities to tracklets.")

    return allframes


def create_velocity_video(video_path: str, tracklets_path: str, velocities=None, dest_folder=None, smooth_factor=1,
                          start_index=0, nframes=None, mask_xy=(0, 0), mask_dimensions=None, show_mask=False,
                          save_as_csv=False, overwrite=False):
    # Only use this function after generating frames for the whole video. Otherwise, a shorter video will be made
    frames_path = video_to_frames(video_path)
    vel_path = dest_folder if dest_folder is not None else os.path.join(os.path.dirname(tracklets_path), "velocities")
    if not os.path.exists(vel_path):
        os.mkdir(vel_path)
    frames = velocities if velocities is not None else get_velocities(tracklets_path, smooth_factor, start_index,
                                                                      nframes, mask_xy, mask_dimensions, save_as_csv)
    vel_directory = os.listdir(vel_path)
    if len([frame for frame in vel_directory if frame.endswith(".png")]) == len(frames) and not overwrite:
        print("Velocities already plotted. Exiting...")
        return

    for frame_num, fishes in tqdm(frames.items(), desc="Plotting velocities..."):
        frame_path = os.path.join(frames_path, f"{frame_num}.png")
        try:
            plot_velocities(frame=frame_path, frame_num=frame_num, fishes=fishes, destfolder=vel_path, show=False,
                            xy=mask_xy, dimensions=mask_dimensions, show_mask=show_mask)
        except Exception as e:
            print(f"Could not plot velocities for {frame_num}. {e}")
            continue

    if overwrite:
        path = os.path.join(vel_path, "velocity.mp4")
        if os.path.exists(path):
            os.remove(path)

    _create_velocity_video(vel_path)


def euclidean_distance(a: list, b: list):
    a_pos = list()
    b_pos = list()
    for i in a:
        a_pos.append(i[0])
        a_pos.append(i[1])
    for i in b:
        b_pos.append(i[0])
        b_pos.append(i[1])
    return math.dist(a_pos, b_pos)


def prettify(a: typing.Dict[typing.AnyStr, Track]):
    for k, v in a.items():
        print(f"{k}-{v.b.id} | Start: {v.start} | End: {v.end} | Track length: {v.length}")


def a_directed_towards_b(a: Fish, b: Fish, threshold=60) -> bool:
    """
        Checks if the velocity of a's front is directed towards b's tail
        **Note that this method is not symmetric, a directed towards b does not imply that b is directed towards a
    """
    theta = math.radians(threshold)
    u = a.vel[0].direction
    v = b.position[-1] - a.position[0]
    v /= np.linalg.norm(v)
    return abs(np.arccos(np.dot(u, v))) < theta


def track_bower_circling(video: str, frames: typing.Dict[typing.AnyStr, typing.Dict[typing.AnyStr, Fish]],
                         proximity: int,
                         head_tail_proximity: int, track_age: int, threshold: int, bower_circling_length: int,
                         extract_clips: bool):
    tracks = {}
    for frame_num, frame in tqdm(frames.items(), desc="Tracking bower circling incidents..."):
        fish_nums = list(frame.keys())
        fishes = list(frame.values())
        # bower circling can only happen between at least two fish
        if len(fishes) < 2:
            continue
        matched = set()
        # check every combination of fish and exit when one pair is made (BC cannot happen between more than two fish)
        # this may be changed later if correct pairs are being missed
        for i in range(len(fishes) - 1):
            if fish_nums[i] in matched:
                continue
            a = fishes[i]
            min_dist = sys.maxsize
            closest_b = -1

            for j in range(i + 1, len(fishes)):
                if fish_nums[j] in matched:  # this fish already has a pair, so skip
                    continue

                b = fishes[j]
                distance = euclidean_distance(a.position, b.position)
                if distance > proximity:  # fish must be close
                    continue

                ahead_btail = np.linalg.norm(a.position[0] - b.position[-1])
                atail_bhead = np.linalg.norm(a.position[-1] - b.position[0])
                # a's head must be close to b's tail and a's tail must be close to b's head
                if ahead_btail > head_tail_proximity or atail_bhead > head_tail_proximity:
                    continue

                # if a velocity magnitude check becomes necessary, place it here

                # a's front should be directed towards b's tail, and b's front should be directed towards a's tail
                if not (a_directed_towards_b(a, b, threshold) and a_directed_towards_b(b, a, threshold)):
                    continue

                # track already exists, so we'll update it if it's not dead
                if tracks.get(a.id) and tracks[a.id].b.id == b.id and not tracks[a.id].is_dead(frame_num, track_age):
                    tracks[a.id].length += (str_to_int(frame_num) - str_to_int(tracks[a.id].end))
                    tracks[a.id].end = frame_num
                    matched.add(a.id)
                    matched.add(b.id)
                    break

                # only add the closest a and b pair if no pair has been made yet
                if distance > min_dist:
                    continue

                min_dist = distance
                closest_b = j

            if closest_b == -1:
                continue
            if not tracks.get(a.id):
                tracks.update({a.id: Track(a=a, b=fishes[closest_b], start=frame_num, end=frame_num, length=1)})
                matched.add(a.id)
                matched.add(fish_nums[closest_b])

    bower_circling_incidents = [track for track in tracks.values() if track.length >= bower_circling_length]
    if len(bower_circling_incidents) == 0:
        print("No bower circling incidents found.")
        return None
    print(bower_circling_incidents)

    width = int(math.log10(len(frames))) + 1

    for incident in bower_circling_incidents:
        for i in range(str_to_int(incident.start), str_to_int(incident.end) + 1):
            frames[f"frame{i:0{width}d}"][incident.a.id].bc = True
            frames[f"frame{i:0{width}d}"][incident.b.id].bc = True

    print(f"Added {len(bower_circling_incidents)} bower circling track(s) to frames data.")

    if not extract_clips:
        return bower_circling_incidents

    if video is None or len(video) == 0 or not os.path.exists(video):
        raise TypeError("Video path cannot be empty.")

    output_dir = os.path.join(os.path.dirname(video), "bower-circling-clips")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fps = get_video_fps(video)

    for incident in tqdm(bower_circling_incidents, "Extracting bower circling clips..."):
        start = str(timedelta(seconds=(str_to_int(incident.start) / fps)))
        end = str(timedelta(seconds=(str_to_int(incident.end)) / fps))
        length = str(timedelta(seconds=(incident.length / fps)))
        out_file = os.path.join(output_dir, f"{start[:10]}-{end[:10]}.mp4")
        s.call(['ffmpeg', '-ss', start, '-accurate_seek', '-i', video, '-t', length, '-c:v', 'libx264',
                '-c:a', 'aac', out_file])

    return bower_circling_incidents
