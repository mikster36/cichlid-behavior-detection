import math
import os
import pickle
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


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

    def avg_vel(self):
        magnitude = 0
        direction = np.ndarray(2,)
        for v in self.vel:
            magnitude += v.magnitude / len(self.vel)
            direction += v.direction / len(self.vel) if not np.isnan(v.direction[0]) else 0
        return Vel(direction=direction, magnitude=magnitude)


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

    def is_dead(self, frame: str, t: int):
        curr_frame = str_to_int(frame)
        prev_frame = str_to_int(self.end)
        return curr_frame - prev_frame > t


def get_velocities(tracklets_path: str, smooth_factor=1, mask_xy=(0, 0), mask_dimensions=None,
                   save_as_csv=False) -> typing.Dict[typing.AnyStr, typing.Dict]:
    # check if velocities have already been calculated
    vel_pick = os.path.join(os.path.dirname(tracklets_path), f"{Path(tracklets_path).stem}_velocities.pickle")
    if os.path.exists(vel_pick):
        with open(vel_pick, 'rb') as handle:
            print(f"Velocities retrieved from {vel_pick}")
            return pickle.load(handle)

    if tracklets_path.endswith('h5'):
        tracklets: pd.DataFrame = pd.DataFrame(pd.read_hdf(tracklets_path))
        tracklets.columns = tracklets.columns.droplevel(0)
    else:
        tracklets: pd.DataFrame = pd.DataFrame(pd.read_csv(tracklets_path, header=[0, 1, 2], index_col=0))

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

    with open(vel_pick, 'wb') as handle:
        pickle.dump(allframes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return allframes


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


def str_to_int(s: str):
    out = str()
    for c in s:
        if c.isdigit():
            out += c
    return int(out)


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


def point_in_focus(x, y, mask_x, mask_y, width, height) -> bool:
    return mask_x <= x <= mask_x + width and mask_y <= y <= mask_y + height


def rotate(dx, dy, angle) -> typing.Tuple[float, float]:
    rad = math.radians(angle)
    ndx = dx * math.cos(rad) + dy * math.sin(rad)
    ndy = -dx * math.sin(rad) + dy * math.cos(rad)
    return ndx, ndy


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


def prettify(a: typing.List[Track]):
    for track in a:
        print(f"{track.a.id}-{track.b.id} | Start: {track.start} | End: {track.end} | "
              f"Track length: {str_to_int(track.end) - str_to_int(track.start) + 1}")


def a_directed_towards_b(a: Fish, b: Fish, theta=1.0472) -> bool:
    """
        Checks if the velocity of a's front is directed towards b's tail
        **Note that this method is not symmetric, a directed towards b does not imply that b is directed towards a
    """
    u = a.vel[0].direction
    v = b.position[-1] - a.position[0]
    if np.isnan(u).any() or np.isnan(v).any():
        return False
    v = v.astype(dtype=np.dtype('float32'))
    v /= np.linalg.norm(v)
    return abs(np.arccos(np.dot(u, v))) < theta
