import deeplabcut as dlc
import yaml
import re

shuffles = [1, 2]
training_iterations = [100000, 150000]

project_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26"
config_path = f"{project_path}/config.yaml"
password_path = "/home/bree_student/anaconda3/envs/dlc/cichlid-behavior-detection/password.yaml"
with open(password_path, 'r') as f:
    sudo_password = yaml.safe_load(f)['password']
sudo_password += '\n'


def kill_and_reset():
    import subprocess
    # via chatGPT
    # Step 1: Run nvidia-smi to list GPU processes and their memory usage
    nvidia_smi_output = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout

    # Step 2: Parse the output to identify the process with the highest GPU memory usage
    # Assuming the output format of nvidia-smi is consistent
    process_lines = [line for line in nvidia_smi_output.split('\n') if 'python' in line]
    if process_lines:
        result = re.split(r'\D+', process_lines[0])
        result = [s for s in result if s]
        pid = result[1]
        print(pid)
    else:
        print("No GPU processes found.")
        exit(1)
    # Step 3: Use subprocess to kill the identified process

    try:
        command = ['sudo', 'kill', '-9', pid]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)
        process.stdin.write(sudo_password)
        process.stdin.flush()
        print(f"Killed process with PID {pid}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to kill process with PID {pid}. Error: {e}")

"""
for shuffle, maxiters in zip(shuffles, training_iterations):
    dlc.train_network(config=config_path, shuffle=shuffle, maxiters=maxiters)
    kill_and_reset()
"""