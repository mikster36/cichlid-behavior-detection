from behavior_detection.BehavioralVideo import BehavioralVideo
import behavior_detection.bower_circling as bc


def main():
    config_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/config.yaml"
    video_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc36_2_Tk3_030320/0001_vidshort.mp4"
    tracklets = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc36_2_Tk3_030320/0001_vidshortDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_el.h5"
    behavioralClip = BehavioralVideo(video_path, config=config_path, shuffle=4, tracklets=tracklets)
    behavioralClip.calculate_velocities(smooth_factor=7)
    behavioralClip.check_bower_circling(threshold=90)


if __name__ == "__main__":
    main()
