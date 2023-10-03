import os
import pickle
import pandas as pd
import numpy as np
import cv2


filepath_pickle = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_assemblies.pickle"
video = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_el_filtered_id_labeled.mp4"
out = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/bower_circling/labeled-frames"

data_pickle = pd.read_pickle(filepath_pickle)

def read_video(video: str, output: str):
    vid = cv2.VideoCapture(video)
    success, image = vid.read()
    count = 0
    while success:
      cv2.imwrite(f"{os.path.join(output, f'frame{count}.png')}", image)
      success,image = vid.read()
      print('Read a new frame: ', success)
      count += 1


def show_nframes(frames: str, n: int):
    for i in range(n):
        image = cv2.imread(f"{os.path.join(frames, f'frame{i}.png')}")
        window_name = f'frame{i}'
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()

def get_centroid(xy_coords: list[tuple]):
    x_sum, y_sum = 0, 0
    for i in xy_coords:
        x_sum += i[0]
        y_sum += i[1]
    return (x_sum / len(xy_coords), y_sum / len(xy_coords))

frame = data_pickle[70]
for individual in frame:
    # x-y tuples for nose, lefteye, righteye, and spine1
    # the centre of this cluster is our approximation for head location
    front_cluster = [(individual[i][0], individual[i][1]) for i in range(4)]
    head = get_centroid(front_cluster)
    tail = (individual[-1][0], individual[-1][1])
    print(head, tail)
