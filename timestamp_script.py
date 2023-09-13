
import json
import os
from moviepy.editor import VideoFileClip

# Load JSON file
with open('../metadata.json', 'r') as f:
    data = json.load(f)

# Load MP4 file
video = VideoFileClip('/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/0002_vid.mp4')

# Iterate through metadata to create video snippets
for name in data['metadata']:
    snippet = data['metadata'][name]
    print(snippet)
    start_time, end_time = snippet['z']
    special_var = snippet['av']['1']
   
    # Create subfolder if it doesn't exist
    subfolder_path = os.path.join('..', special_var)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
   
    # Create snippet video
    snippet_video = video.subclip(start_time, end_time)
   
    # Save snippet video
    snippet_filename = f"{start_time}_{end_time}.mp4"
    snippet_path = os.path.join(subfolder_path, snippet_filename)
    snippet_video.write_videofile(snippet_path)