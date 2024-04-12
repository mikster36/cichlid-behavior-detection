from behavior_detection.misc.analyse_videos import analyse_videos


def main():
    from behavior_detection.misc.train_network import kill_and_reset
    # train ben's model
    config_path = r"/media/bree_student/Elements/DLC_models/dlc_model-student-2024-01-30/config.yaml"
    import deeplabcut as dlc
    dlc.train_network(config=config_path, maxiters=100000, displayiters=1000, saveiters=15000, allow_growth=True)
    kill_and_reset()


    config_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/config.yaml"
    video_path = [r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc40_2_Tk3_030920",
                  r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc41_2_Tk9_030920/0002_vid.mp4",
                  r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc43_11_Tk41_060220/0001_vid.mp4",
                  r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc44_7_Tk65_050720/0002_vid.mp4",
                  r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc45_7_Tk47_050720/0002_vid.mp4",
                  r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc55_2_Tk47_051220/0002_vid.mp4",
                  r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc56_2_Tk65_051220/0002_vid.mp4",
                  r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc59_4_Tk61_060220/0001_vid.mp4",
                  r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc62_3_Tk65_060220/0001_vid.mp4",
                  r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc63_1_Tk9_060220/0001_vid.mp4"]
    for video in video_path:
        try:
            analyse_videos(config_path=config_path, videos=[video], shuffle=4)
        except Exception:
            kill_and_reset()
            continue


if __name__ == "__main__":
    main()
