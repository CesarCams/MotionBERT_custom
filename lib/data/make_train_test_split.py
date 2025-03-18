import os
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Path to the processed videos directory
video_dir = '/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionBERT/lib/data/processed_videos'

# Dictionary to hold video file names by class
videos_by_class = defaultdict(list)
# List the folders in the video directory
folders = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
#print("Folders in video_dir:", folders)
for folder in folders:
    folder_path = os.path.join(video_dir, folder)
    #print(f"Folder path: {folder_path}")
    subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    #print(f"Subfolders in {folder}:", subfolders)
    for subfolder in subfolders:
        class_name = folder
        folder_name = subfolder
        videos_by_class[class_name].append(folder_name)

# Split the videos into train and test sets while respecting class proportions
train_videos = []
test_videos = []

for class_name, videos in videos_by_class.items():
    train, test = train_test_split(videos, test_size=0.5, random_state=42)
    train_videos.extend(train)
    test_videos.extend(test)


print(f"Train videos: {len(train_videos)}")
print(f"Test videos: {len(test_videos)}")

# Ensure the splits directory exists
splits_dir = 'splits'
os.makedirs(splits_dir, exist_ok=True)

# Write the train video names to a .txt file
with open(os.path.join(splits_dir, 'train_list_0.5.txt'), 'w') as train_file:
    for video in train_videos:
        train_file.write(f"{video}\n")

# Write the test video names to a .txt file
with open(os.path.join(splits_dir, 'test_list_0.5.txt'), 'w') as test_file:
    for video in test_videos:
        test_file.write(f"{video}\n")

print("Train and test video lists have been created.")
