#!/usr/bin/env python
"""
    This script extracts potential 'bower circling' clips from a video
    @author: mikster36
    @date: 10/2/23
"""
import sys

import matplotlib

from behavior_detection.misc.tracking import *
from behavior_detection.misc.tracking import str_to_int
from behavior_detection.misc.video_auxiliary import _extract_clips

matplotlib.use("TKAgg")
np.seterr(divide='ignore', invalid='ignore')


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


def track_bower_circling(video: str, frames: typing.Dict[typing.AnyStr, typing.Dict[typing.AnyStr, Fish]],
                         proximity: int,
                         head_tail_proximity: int, track_age: int, threshold: int, bower_circling_length: int,
                         extract_clips: bool, debug=False):
    theta = math.radians(threshold)
    tracks = {}
    for frame_num, frame in tqdm(frames.items(), desc="Tracking bower circling incidents..."):
        fish_nums = list(frame.keys())
        fishes = list(frame.values())
        # bower circling generally only happens when 2 fish are on the screen
        if len(fishes) < 2 or len(fishes) > 3:
            continue
        matched = set()
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
                if distance > proximity or distance < 50:  # fish must be close, but not too close
                    if debug:
                        print(f"Failed basic proximity check at {fish_nums[i]}-{fish_nums[j]} in {frame_num}. "
                              f"Distance: {distance}.")
                    continue

                ahead_btail = np.linalg.norm(a.position[0] - b.position[-1])
                atail_bhead = np.linalg.norm(a.position[-1] - b.position[0])
                # a's head must be close to b's tail and a's tail must be close to b's head
                if ahead_btail > head_tail_proximity or atail_bhead > head_tail_proximity:
                    if debug:
                        print(f"Failed head-tail proximity check at {fish_nums[i]}-{fish_nums[j]} in {frame_num}. "
                              f"Ahead_btail distance: {ahead_btail}. Atail_bhead distance: {atail_bhead}.")
                    continue

                # if a velocity magnitude check becomes necessary, place it here

                # a's front should be directed towards b's tail, and b's front should be directed towards a's tail
                if not (a_directed_towards_b(a, b, theta) and a_directed_towards_b(b, a, theta)):
                    if debug:
                        print(f"Failed 'a directed towards b' check at {fish_nums[i]}-{fish_nums[j]} in {frame_num}.")
                        print(f"A velocity: {a.vel}. B velocity: {b.vel}")
                    continue

                # track does not exist yet
                if not tracks.get(a.id):
                    # only add the closest a and b pair if no pair has been made yet
                    if distance > min_dist:
                        continue

                    min_dist = distance
                    closest_b = j
                    continue
                track = tracks.get(a.id)[-1]  # most recent track
                # most recent track is with a different b fish, so create a new track
                if track.b.id != b.id:
                    tracks[a.id].append(Track(a=a, b=b, start=frame_num, end=frame_num))
                    matched.add(a.id)
                    matched.add(b.id)
                    break
                # most recent track matches this b fish and is not dead, so update it
                if not track.is_dead(frame_num, track_age):
                    track.end = frame_num
                    matched.add(a.id)
                    matched.add(b.id)
                    break
                # most recent track matches this b fish and is dead, so create a new track
                tracks[a.id].append(Track(a=a, b=b, start=frame_num, end=frame_num))
                matched.add(a.id)
                matched.add(b.id)

            # no match found
            if closest_b == -1:
                continue
            # track does not exist yet
            if not tracks.get(a.id):
                tracks.update({a.id: [Track(a=a, b=fishes[closest_b], start=frame_num, end=frame_num)]})
                matched.add(a.id)
                matched.add(fish_nums[closest_b])

    bower_circling_incidents = list()
    for tracks in tracks.values():
        for track in tracks:
            if str_to_int(track.end) - str_to_int(track.start) + 1 >= bower_circling_length:
                bower_circling_incidents.append(track)

    if len(bower_circling_incidents) == 0:
        print("No bower circling incidents found.")
        return None

    prettify(bower_circling_incidents)

    # this portion is only necessary if we wanted to maintain bower circling metadata for each frame
    """
    width = int(math.log10(get_video_length(video) * get_video_fps(video))) + 1
    
        for incident in bower_circling_incidents:
        for i in range(str_to_int(incident.start), str_to_int(incident.end) + 1):
            frames[f"frame{i:0{width}d}"][incident.a.id].bc = True
            frames[f"frame{i:0{width}d}"][incident.b.id].bc = True
    """

    print(f"Added {len(bower_circling_incidents)} bower circling track(s) to frames data.")

    if not extract_clips:
        return bower_circling_incidents

    _extract_clips(bower_circling_incidents, video)

    return bower_circling_incidents


