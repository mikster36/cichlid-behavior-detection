from behavior_detection.misc.dropbox_handling import get_clips_from_clustering_data

def main():
	trial = "/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/MC_singlenuc29_3_Tk9_030320"
	get_clips_from_clustering_data(trial=trial, behavior='s')



if __name__ == "__main__":
	main()