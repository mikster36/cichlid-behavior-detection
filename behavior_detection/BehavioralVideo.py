import typing

from pathlib import Path
from os import path as path
import pickle

import behavior_detection.bower_circling as bc
import behavior_detection.misc.maskGUI as mask
import behavior_detection.misc.tracking
import behavior_detection.misc.video_auxiliary


class BehavioralVideo:
    """
    A class to represent a video containing any amount of behavioral incidents
    That behavior can be one of the following:
    Aggressive:
        - chase / flee
    Flirting:
        - display, quiver, bower-circling / spawning, lead

    Attributes
    ----------
    video: str
        the path to a video
    tracklets_path: str
        the path to the tracklets file associated with a clip (file that ends with *_{el/bo/sk}.h5)
    frames: list[dict]
        a list of every frame in the clip, with every fish's location, velocity, and behavior at each frame
    mask_xy: tuple[int, int]
        the upper left coordinates of a rectangle mask to define where fish are in bounds
    mask_dimensions: tuple[int, int]
        the width and height of the mask to define fish in bounds as a tuple (width, height)

    Methods
    -------
    calculate_velocities(...)
        gets the velocities in a clip
    create_velocity_video(...)
        makes a video with the velocities of each fish annotated

    Mask usage:
            -----------------------
            |         3           |
            |   - - - - - -       |
            |   | 1       |       |
            |   |      2  |   4   |
            |   - - - - - -       |
            |            5        |
            -----------------------
    1 and 2 are in bounds, whereas 3, 4, and 5 are out of bounds and ignored
    * this will be prompted via a GUI
    """

    def __init__(self, video_path: str, config=None, shuffle=None, tracklets_path=None, headless=False):
        self.video = video_path
        if tracklets_path is None:
            # Analyse video if it hasn't been analysed
            import behavior_detection.misc.analyse_videos as analysis
            import os, glob
            analysis.analyse_videos(config, [video_path], shuffle=shuffle)
            self.tracklets_path = glob.glob(os.path.dirname(video_path) + '*filtered.h5')
            for file in os.listdir(os.path.dirname(video_path)):
                if file.endswith('filtered.h5'):
                    self.tracklets_path = os.path.join(os.path.dirname(video_path), file)
            if self.tracklets_path is None:
                print("Tracklets (*_filtered.h5 file) not found.")
        else:
            self.tracklets_path = tracklets_path

        vel_pick = path.join(path.dirname(tracklets_path), f"{Path(tracklets_path).stem}_velocities.pickle")
        if path.exists(vel_pick):
            with open(vel_pick, 'rb') as handle:
                print(f"Velocities retrieved from {vel_pick}")
                self.frames = pickle.load(handle)
        else:
            self.frames = None
        if headless and self.frames is None:
            x = int(input("X:"))
            y = int(input("Y:"))
            w = int(input("Width:"))
            l = int(input("Length: "))
            self.mask_xy = (x, y)
            self.mask_dimensions = (w, l)
        else:
            if self.frames is None:
                self.mask_xy, self.mask_dimensions = mask.get_mask(video_path)
            else:
                self.mask_xy, self.mask_dimensions = None, None

    def calculate_velocities(self,
                             smooth_factor=1,
                             save_as_csv=False) -> typing.Dict[typing.AnyStr, typing.Dict]:
        """
        Gets the velocity of all fish across all frames where each fish is seen in all {smooth_factor + 1} frames

        Args:
            smooth_factor: the distance of frames used to calculate velocity. By default, this is 1, which provides no
            smoothing. A generally good smooth factor for a 30fps video is between 6-8. Increase this number if your
            video frame rate is higher
            save_as_csv: whether to save the calculated velocities in a csv file. The csv will be stored in the same
            directory as the tracklets file, following the naming convention of the tracklets file


        Returns:
            dict[str: dict]: a dictionary where the keys are frame numbers and its values are a dictionary
            with fish numbers as keys and Fish objects as values (a Fish object has position and velocity)
        """
        self.frames = behavior_detection.misc.tracking.get_velocities(tracklets_path=self.tracklets_path, smooth_factor=smooth_factor,
                                                                              mask_xy=self.mask_xy,
                                                                              mask_dimensions=self.mask_dimensions, save_as_csv=save_as_csv)
        return self.frames

    def create_velocity_video(self,
                              dest_folder=None,
                              smooth_factor=1,
                              start_index=0,
                              nframes=None,
                              show_mask=False,
                              save_as_csv=False,
                              overwrite=False):
        """
        Overlays the velocity and coordinates of a fish's centre, middle, and tail onto a video of fish

        Args:
             dest_folder: where the velocity video should be stored
             smooth_factor: the distance of frames used to calculate velocity. By default, this is 1, which provides no
             smoothing. A general, decent smooth factor for a 30fps video is between 6-8. Increase this number if your
             video frame rate is higher
             start_index: what frame number to start at. By default, this is 0
             nframes: the number of frames to include in the video. By default, this is the length of the video (in
             number of frames)
             show_mask: whether to include the in bounds area in the video
             save_as_csv: whether to save the calculated velocities in a csv file. The csv will be stored in the same
             directory as the tracklets file, following the naming convention of the tracklets file
             overwrite: whether to overwrite the existing velocity video (if there is one)
        """
        self.frames = self.calculate_velocities(smooth_factor=smooth_factor, save_as_csv=save_as_csv)
        behavior_detection.misc.video_auxiliary.create_velocity_video(video_path=self.video, tracklets_path=self.tracklets_path, velocities=self.frames, dest_folder=dest_folder, smooth_factor=smooth_factor,
                                                                      start_index=start_index, nframes=nframes, mask_xy=self.mask_xy,
                                                                      mask_dimensions=self.mask_dimensions, show_mask=show_mask, overwrite=overwrite)

    def check_bower_circling(self,
                             proximity=250,
                             head_tail_proximity=180,
                             threshold=60,
                             track_age=18,
                             bower_circling_length=30,
                             extract_clips=True,
                             buffer = 0):
        """
        Checks to see if bower circling occurs in this clip

        Args:
            proximity: int
                how close (in px) a pair of fish should be for potential bower circling to occur
            head_tail_proximity: int
                how close a fish's tail must be to the head of another fish (and vice versa)
                for potential bower circling to occur
            threshold: int
                the margin of error (in degrees) of which the head of one fish can point towards the tail of
                another fish. This is used to determine if a fish's head is reasonably directed toward another fish's tail
            track_age: int
                how many frames of unmet above criteria before considering a track 'dead'. This should be
                smaller if your velocities and position data is robust
            bower_circling_length: int
                the minimum length a track can be before it is considered a bower circling incident
            extract_clips: bool
                whether to save bower circling clips
        """
        if self.frames is None:
            raise ValueError("Make sure to run calculate_velocities() or create_velocity_video() before attempting to "
                             "track bower circling.")
        bc.track_bower_circling(self.video, self.frames, proximity, head_tail_proximity, track_age, threshold,
                                bower_circling_length, extract_clips, buffer)

    def set(self, **kwargs):
        """
        Sets any field. Use with caution to avoid unexpected behaviour
        """
        for key in kwargs:
            if key == 'video_path':
                self.video = kwargs['video_path']
            elif key == 'tracklets_path':
                self.tracklets_path = kwargs['tracklets_path']
            elif key == 'velocities':
                self.frames = kwargs['velocities']
