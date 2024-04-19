import json
import os
from moviepy.editor import VideoFileClip

'''
Extracts clips from json file created using VGG Annotator.

IMPORTANT!
- All videos must be named '0001_vid.mp4'
- Video directory must be formated as seen below:

videos
└── MC_singlenuc36_2_Tk3_030320
    ├── 0001_vid.mp4
    ├── metadata.json
└── MC_singlenuc35_2_Tk3_030320
    ├── 0001_vid.mp4
    ├── metadata.json
└── MC_singlenuc29_6_Tk3_030320
    ├── 0001_vid.mp4
    ├── metadata.json
└── ...

'''
def extract_clips(video_dir):
    if not os.path.exists(video_dir):
        print("Directory does not exist!")
        return

    # Iterate through trial folders
    for trial in os.listdir(video_dir):
        print(f"Currently working on: {trial}")

        # Load JSON file
        with open(os.path.join(video_dir, trial, 'metadata.json'), 'r') as f:
            data = json.load(f)

        # Load MP4 file
        video = VideoFileClip(os.path.join(video_dir, trial, '0001_vid.mp4'))

        # Iterate through metadata to create video snippets
        for name in data['metadata']:
            snippet = data['metadata'][name]

            try:
                start_time, end_time = [round(time, 2) for time in snippet['z']]
                behavior = snippet['av']['1']
        
                # Create subfolder if it doesn't exist
                subfolder_path = os.path.join(os.path.join(video_dir, trial), behavior)
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
            
                # Create snippet video
                snippet_video = video.subclip(start_time, end_time)
            
                # Save snippet video
                snippet_filename = f"{start_time}_{end_time}.mp4"
                snippet_path = os.path.join(subfolder_path, snippet_filename)
                snippet_video.write_videofile(snippet_path)
            except:
                print(f"The clip starting at {snippet['z'][0]} is missing an end time. Enter 'y' to continue program execution and skip clip, or enter 'n' to terminate program execution")
                res = input("Enter 'y' or 'n': ")
                while(res not in ['y', 'n']):
                    res = input("Enter 'y' or 'n': ")
                if res == 'n':
                    return

            

def main():
    # Enter video directory
    video_dir = input("Enter video directory absolute path: ")
    extract_clips(video_dir)

if __name__ == '__main__':
    main()