from behavior_detection.BehavioralClip import BehavioralClip
import behavior_detection.bower_circling as bc


def main():
    config_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/config.yaml"
    video_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/bower_circling/bower_circling.mp4"
    tracklets = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_el_filtered.h5"
    behavioralClip = BehavioralClip(video_path, config=config_path, shuffle=4)
    behavioralClip.calculate_velocities(smooth_factor=7)
    behavioralClip.check_bower_circling(threshold=90)

    behavioralClip.create_velocity_video(fps=29, overwrite=True, show_mask=False)
    #bc.extract_bower_circling_clips(video_path, 29, behavioralClip.frames)


if __name__ == "__main__":
    main()
