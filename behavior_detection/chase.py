import math
import typing

import sys
from tqdm import tqdm

from behavior_detection.BehavioralVideo import BehavioralVideo
from behavior_detection.misc.tracking import str_to_int, euclidean_distance, prettify, a_directed_towards_b, Track
from behavior_detection.misc.tracking import Fish
from behavior_detection.misc.video_auxiliary import extract_incidents


def track_chasing(video: str, frames: typing.Dict[typing.AnyStr, typing.Dict[typing.AnyStr, Fish]],
                  track_age: int, threshold: int, chase_length: int, extract_clips: bool, debug=False, chase_speed=15):
    theta = math.radians(threshold)
    tracks = {}
    for frame_num, frame in tqdm(frames.items(), desc="Tracking bower circling incidents..."):
        fish_nums = list(frame.keys())
        fishes = list(frame.values())
        # need at least two fish to chase each other
        if len(fishes) < 2:
            continue
        matched = set()
        for i in range(len(fishes) - 1):
            if fish_nums[i] in matched:
                continue
            a = fishes[i]
            min_dist = sys.maxsize
            closest_b = -1

            for j in range(i + 1, len(fishes)):
                # fish already has a pair so skip it
                if fish_nums[j] in matched:
                    continue

                b = fishes[j]
                distance = euclidean_distance(a.position, b.position)

                # a is the chaser, b is being chased
                # a must be chasing b
                if not a_directed_towards_b(a, b, theta):
                    if debug:
                        print(f"Failed 'a directed towards b' check at {fish_nums[i]}-{fish_nums[j]} in {frame_num}.")
                        print(f"A velocity: {a.vel}. B velocity: {b.vel}")
                    continue

                # a is not moving fast enough
                if a.avg_vel().magnitude < chase_speed:
                    if debug:
                        print(f"Failed speed check at {fish_nums[i]} in {frame_num}.")
                        print(f"A's velocity: {a.vel}")
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

    chase_incidents = list()
    for tracks in tracks.values():
        for track in tracks:
            if str_to_int(track.end) - str_to_int(track.start) + 1 >= chase_length:
                chase_incidents.append(track)

    if len(chase_incidents) == 0:
        print("No chase incidents were found.")
        return None

    prettify(chase_incidents)

    print(f"Added {len(chase_incidents)} chase track(s) to frames data.")

    if extract_clips:
        extract_incidents(chase_incidents, video, behavior="chase")

    return chase_incidents



if __name__ == "__main__":
    tracklets = ("/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc36_2_Tk3_030320/"
                 "batches/batch0/0001_vidDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_el_filtered.h5")
    video = "/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc36_2_Tk3_030320/batches/batch0/0001_vid.mp4"
    vid = BehavioralVideo(video_path=video, tracklets_path=tracklets, headless=True)
    track_chasing(vid.video, vid.frames, track_age=18, threshold=90, chase_length=10, extract_clips=True)
