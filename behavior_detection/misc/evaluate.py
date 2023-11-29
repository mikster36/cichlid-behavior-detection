#import deeplabcut
import yaml
import os

project_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26"
config_path = f"{project_path}/config.yaml"


def change_config(NEW_ITERATION):
    old_iteration = -1
    old_path = ""
    flag = True

    for file in os.listdir(project_path):
        if "config" in file:
            with open(f"{project_path}/{file}", "r") as f:
                data = yaml.safe_load(f)
            if str(file) != "config.yaml":
                if NEW_ITERATION == data['iteration']:
                    old_path = f"{project_path}/{file}"
            else:
                if NEW_ITERATION == data['iteration']:
                    flag = False
                    break
                old_iteration = data['iteration']

    if flag:
        os.rename(project_path + "/config.yaml", project_path + f"/config-{old_iteration}.yaml")
        os.rename(old_path, project_path + "/config.yaml")


change_config(7)
