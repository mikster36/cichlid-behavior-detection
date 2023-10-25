from behavior_detection.BehavioralClip import BehavioralClip
from behavior_detection.misc_scripts.analyse_videos import analyse_videos

if __name__ == "__main__":
    config_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/config.yaml"
    video_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/testing/testing0/testing0.mp4"
    tracklets = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/testing/testing0/testing0DLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_el_filtered.h5"
    analyse_videos(config_path, video_path)
    behavioralClip = BehavioralClip(video_path, tracklets)
    behavioralClip.create_velocity_video(fps=29, smooth_factor=7)
