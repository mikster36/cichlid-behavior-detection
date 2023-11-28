import glob
import os


from behavior_detection.BehavioralVideo import BehavioralVideo


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
    config_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/config.yaml"
    batches = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc23_1_Tk33_021220/batches"
    bower_circling_in_batches(config_path, batches, 4)


if __name__ == "__main__":
    main()
