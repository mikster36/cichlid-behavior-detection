import math
import os
import subprocess as s
import typing
from datetime import timedelta

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, patches as patches
from pathlib import Path
from tqdm import tqdm

from behavior_detection.misc.ffmpeg_split import get_video_length
from behavior_detection.misc.tracking import Fish, get_velocities, point_in_focus, str_to_int


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


def create_velocity_video(video_path: str, tracklets_path: str, velocities=None, dest_folder=None, smooth_factor=1,
                          start_index=0, nframes=None, mask_xy=(0, 0), mask_dimensions=None, show_mask=False,
                          save_as_csv=False, overwrite=False):
    # Only use this function after generating frames for the whole video. Otherwise, a shorter video will be made
    frames_path = video_to_frames(video_path)
    vel_path = dest_folder if dest_folder is not None else os.path.join(os.path.dirname(tracklets_path), "velocities")
    if not os.path.exists(vel_path):
        os.mkdir(vel_path)
    frames = velocities if velocities is not None else get_velocities(tracklets_path, smooth_factor, start_index,
                                                                      nframes, mask_xy)
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


# generalise this to work with any kind of behavior, not just bower circling
def extract_incidents(bower_circling_incidents: list, video: str, buffer: int, behavior: str):
    if video is None or len(video) == 0 or not os.path.exists(video):
        raise TypeError("Video path cannot be empty.")

    batch_num = None
    parent = Path(video).parent

    if "batch" in parent.name:
        batch_num = str_to_int(parent.name)

    output_dir = os.path.join(Path(video).parent.absolute() if batch_num
                              else os.path.dirname(video), f"{behavior}-clips")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fps = get_video_fps(video)
    for incident in tqdm(bower_circling_incidents, f"Extracting {behavior} clips..."):
        start_f, end_f = str_to_int(incident.start) - (fps * buffer), str_to_int(incident.end) + (fps * buffer)
        start = str(timedelta(seconds=(start_f / fps)))
        abs_start = start if batch_num is None else str(timedelta(seconds=(start_f / fps) + 3600 * batch_num))
        end = str(timedelta(seconds=(end_f / fps)))
        abs_end = end if batch_num is None else str(timedelta(seconds=(end_f / fps) + 3600 * batch_num))
        length = str(timedelta(seconds=((end_f - start_f + 1) / fps)))
        out_file = os.path.join(output_dir, f"{abs_start[:10]}-{abs_end[:10]}.mp4")
        s.call(['ffmpeg', '-ss', start, '-accurate_seek', '-i', video, '-t', length, '-c:v', 'libx264',
                '-c:a', 'aac', out_file, '-loglevel', 'quiet'])
