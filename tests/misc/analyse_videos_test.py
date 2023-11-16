from behavior_detection.misc.analyse_videos import analyse_videos


def main():
    config_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/config.yaml"
    video_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc36_2_Tk3_030320/0001_vid.mp4"
    analyse_videos(config_path=config_path, videos=[video_path], shuffle=4)


if __name__ == "__main__":
    main()
