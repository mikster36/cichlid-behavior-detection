import behavior_detection.bower_circling as bc


class BehavioralClip:
    """
    A class to represent a clip of a certain behavior
    That behavior can be one of the following:
    Aggressive:
        - chase / flee
    Flirting:
        - display, quiver, bower-circling / spawning, lead

    Attributes
    ----------
    clip: str
        the path to a video of a behavior
    tracklets: str
        the path to the tracklets file associated with a clip (file that ends with *_{el/bo/sk}}.h5)
    velocities: list[dict]
        a list of every frame in the clip, with every fish's location and velocity at each frame

    Methods
    -------
    calculate_velocities(...)
        gets the velocities in a clip
    create_velocity_video(...)
        makes a video with the velocities of each fish annotated
    """

    def __init__(self, clip_path: str, tracklets: str):
        self.clip = clip_path
        self.tracklets = tracklets
        self.velocities = None

    def calculate_velocities(self,
                             smooth_factor=1,
                             mask_xy=(0, 0),
                             mask_dimensions=None,
                             save_as_csv=False) -> list[dict]:
        """
        Gets the velocity of all fish across all frames where each fish is seen in all {smooth_factor + 1} frames

        Args:
            smooth_factor: the distance of frames used to calculate velocity. By default, this is 1, which provides no
            smoothing. A generally good smooth factor for a 30fps video is between 6-8. Increase this number if your
            video frame rate is higher
             mask_xy: the upper left coordinates of a rectangle mask to define where fish are in bounds
             mask_dimensions: the width and height of the mask to define fish in bounds as a tuple (width, height)
             save_as_csv: whether to save the calculated velocities in a csv file. The csv will be stored in the same
             directory as the tracklets file, following the naming convention of the tracklets file

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

        Returns:
            dict[str: dict]: a dictionary where the keys are frame numbers and its values are a dictionary
            with fish numbers as keys and Fish objects as values (a Fish object has position and velocity)
        """
        self.velocities = bc.get_velocities(tracklets_path=self.tracklets, smooth_factor=smooth_factor, mask_xy=mask_xy,
                                            mask_dimensions=mask_dimensions, save_as_csv=save_as_csv)
        return self.velocities

    def create_velocity_video(self,
                              fps: int,
                              dest_folder=None,
                              smooth_factor=1,
                              start_index=0,
                              nframes=None,
                              mask_xy=(0, 0),
                              mask_dimensions=None,
                              show_mask=False,
                              save_as_csv=False,
                              overwrite=False):
        """
        Overlays the velocity and coordinates of a fish's centre, middle, and tail onto a video of fish

        Args:
             fps: frame rate for the video (this should be the same as the frame rate of the input video)
             dest_folder: where the velocity video should be stored
             smooth_factor: the distance of frames used to calculate velocity. By default, this is 1, which provides no
             smoothing. A general, decent smooth factor for a 30fps video is between 6-8. Increase this number if your
             video frame rate is higher
             start_index: what frame number to start at. By default, this is 0
             nframes: the number of frames to include in the video. By default, this is the length of the video (in
             number of frames)
             mask_xy: the upper left coordinates of a rectangle mask to define where fish are in bounds
             mask_dimensions: the width and height of the mask to define fish in bounds as a tuple (width, height)
             show_mask: whether to include the in bounds area in the video
             save_as_csv: whether to save the calculated velocities in a csv file. The csv will be stored in the same
             directory as the tracklets file, following the naming convention of the tracklets file
             overwrite: whether to overwrite the existing velocity video (if there is one)
        """
        if self.velocities is None:
            self.velocities = self.calculate_velocities(smooth_factor=smooth_factor, mask_xy=mask_xy,
                                                        mask_dimensions=mask_dimensions, save_as_csv=save_as_csv)
        bc.create_velocity_video(video_path=self.clip, tracklets_path=self.tracklets, fps=fps,
                                 velocities=self.velocities, dest_folder=dest_folder, smooth_factor=smooth_factor,
                                 start_index=start_index, nframes=nframes, mask_xy=mask_xy,
                                 mask_dimensions=mask_dimensions, show_mask=show_mask, overwrite=overwrite)
