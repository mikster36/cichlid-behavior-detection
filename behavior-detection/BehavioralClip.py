import bower_circling as bc


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
        the path to the tracklets file associated with a clip (file that ends with *_el.h5)
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
                              save_as_csv=False):
        bc.create_velocity_video(video_path=self.clip, tracklets_path=self.tracklets, fps=fps,
                                 velocities=self.velocities, dest_folder=dest_folder, smooth_factor=smooth_factor,
                                 start_index=start_index, nframes=nframes, mask_xy=mask_xy,
                                 mask_dimensions=mask_dimensions, show_mask=show_mask, save_as_csv=save_as_csv)



