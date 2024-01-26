import os
import subprocess


def upload_to_dropbox(local_folder, remote_folder):
    path = local_folder
    for trial in os.listdir(local_folder):
        bc_clips = os.path.join(path, trial, "bower-circling-clips")
        if not os.path.exists(bc_clips):
            print(f"No bower circling clips found in {trial}")
            continue

        remote_path = os.path.join(remote_folder, trial)
        result = subprocess.run(['rclone', 'lsf', 'dropbox:' + remote_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Folder '{trial}' already exists in Dropbox.")
        else:
            print(f"Creating folder '{trial}' in Dropbox...")
            create_folder_cmd = ['rclone', 'mkdir', 'dropbox:' + remote_path]
            subprocess.run(create_folder_cmd)

        print(f"Uploading {trial} to Dropbox...")
        subprocess.run(['rclone', 'copy', bc_clips, 'dropbox:' + remote_path])


local = '/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/trials/'
remote = 'DLC_annotations/behavior_analysis_output/Bower-circling/'

if __name__ == "__main__":
    upload_to_dropbox(local, remote)
