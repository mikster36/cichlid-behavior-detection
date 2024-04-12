import glob
import os


from behavior_detection.BehavioralVideo import BehavioralVideo
from behavior_detection.misc import dropbox_handling


def bower_circling_in_batches(config: str, batches: str, shuffle=1):
    for batch in os.listdir(batches):
        curr_dir = os.path.join(batches, batch)
        vid_name = glob.glob(os.path.join(curr_dir, '*.mp4'))[0]
        vid_path = os.path.join(curr_dir, vid_name)
        tracklets_path = glob.glob(os.path.join(curr_dir, '*filtered.h5'))[0]
        vid = BehavioralVideo(video_path=vid_path, config=config, shuffle=shuffle, tracklets=tracklets_path)
        vid.calculate_velocities()
        vid.check_bower_circling(threshold=90, extract_clips=True, bower_circling_length=32)
        print(f"Successfully extracted bower circling clips for {batch}")



def main():
    local = '/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/'
    remote = 'DLC_annotations/behavior_analysis_output/Bower-circling/'
    config_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/config.yaml"
    vid_path = '/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc37_2_Tk17_030320'
    dropbox_handling.get_clips(vid_path, config_path)
    dropbox_handling.upload_to_dropbox(local, remote)


if __name__ == "__main__":
    main()
