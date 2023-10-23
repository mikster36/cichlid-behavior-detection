from behavior_detection.BehavioralClip import BehavioralClip
#from behavior_detection.misc_scripts.analyse_videos import *
#from behavior_detection.misc_scripts.analyse_videos import analyse_videos

if __name__ == "__main__":
    config_path = r"C:\Users\michael\dlc_model-student-2023-07-26\config.yaml"
    video_path = r"C:\Users\michael\dlc_model-student-2023-07-26\videos\0001_vid.mp4"
    tracklets = r"C:\Users\michael\dlc_model-student-2023-07-26\videos\0001_vidDLC_dlcrnetms5_dlc_model07-26-2023shuffle4_100000_el_filtered.h5"
    behavioralClip = BehavioralClip(video_path, tracklets)
    behavioralClip.create_velocity_video(fps=29, smooth_factor=7)
