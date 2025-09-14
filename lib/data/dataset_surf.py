import torch
import json
import pickle
import numpy as np
from torch.utils.data import Dataset

class SurfActionDataset(Dataset):
    def __init__(self, data_root, split_list, clip_len=243):
        """
        data_root: folder containing .pkl and .json files
        split_list: list of video names to include (train or val split)
        """
        self.data_root = data_root
        self.video_paths = split_list
        self.clip_len = clip_len
        self.samples = self.load_all_data()

    def load_all_data(self):
        all_data = []
        for video_path in self.video_paths:
            class_name = video_path.split('_')[0]
            dir_path = f"{self.data_root}/{class_name}/{video_path}"
            #dir_path = f"{self.data_root}/{video_path}"
            json_path = f"{dir_path}/keypoints_xyz.json"
            #pkl_path = f"{dir_path}/input_2D/keypoints.npz"
            # Load 3D keypoints from JSON
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            frames = json_data["frames"]
            label_name = json_data["label"]
            label_id = self.label_to_index(label_name)

            #npz_data = np.load(pkl_path)
            # If your data is under a named key:

            #kp2d = npz_data['reconstruction'][0]

            #T = min(len(frames), len(kp2d))
            T = len(frames)
            #print(f"Frames: {len(frames)}, KP2D: {len(kp2d[0])}")

            #print(f"Clip len: {self.clip_len}")
            # if T < 5:
            #    continue  # Skip short clips

            # # Sample uniformly
            # for start in range(0, T - self.clip_len + 1, self.clip_len):
            # if T < self.clip_len:
            #     num_pad = self.clip_len - T
            #     kp2d = np.concatenate([kp2d, np.repeat(kp2d[-1][None, :, :], num_pad, axis=0)], axis=0)
            #     frames += [frames[-1]] * num_pad
            

            pose2d_clip = []
            pose3d_clip = []

            if T >= self.clip_len:
                selected_indices = np.linspace(0, T - 1, self.clip_len).astype(int)
            else:
                # Use all available frames and repeat last frame as needed
                selected_indices = list(range(T)) + [T - 1] * (self.clip_len - T)

            for i in selected_indices:
                # 2D
                #kp_2d = kp2d[i]  # (17, 3)
                #print(f"KP2D shape: {kp_2d.shape}")
                
                #kp_2d = np.pad(kp_2d, ((0,0),(0,1)))  # to (17,3)

                # 3D
                frame = frames[i]["keypoints"]
                kp_3d = np.array([[frame[str(j)]["x"],
                                frame[str(j)]["y"],
                                frame[str(j)]["z"]] for j in range(17)])  # (17,3)

                #pose2d_clip.append(kp_2d)
                pose3d_clip.append(kp_3d)

            #pose2d_clip = np.stack(pose2d_clip)  # (T, 17, 3)
            pose3d_clip = np.stack(pose3d_clip)  # (T, 17, 3)
            dummy_person = np.zeros_like(pose3d_clip) # (T, 17, 3)
            combined = np.stack([pose3d_clip, dummy_person], axis=0)  # (2, T, 17, 3)
            
            all_data.append((combined, label_id))

        print(f"Loaded {len(all_data)} samples")
        return all_data

    def label_to_index(self, label):
        label_map = {
            "360": 0,
            "cutback-frontside": 1,
            "roller": 2,
            "take-off": 3
        }
        return label_map[label]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_tensor, label = self.samples[idx]
        return torch.tensor(input_tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class SurfActionDatasetV2(Dataset):
    def __init__(self, json_path, split_list, clip_len=50, grouping=None):
        """
        json_path: path to dataset.json
        split_list: list of sample indices (train/val/test split)
        clip_len: number of frames per clip
        grouping: dict mapping fine labels -> grouped labels (optional)
        """
        self.json_path = json_path
        self.clip_len = clip_len
        #self.grouping = grouping or {}
        self.samples = self.load_split_data(split_list)


    def load_split_data(self, split_list):
        with open(self.json_path, "r") as f:
            dataset = json.load(f)

        all_data = []
        for idx in split_list:
            sample = dataset["samples"][idx]
            label_name = sample["label"]
            #print(label_name)

            # Apply grouping if provided
            # if label_name in self.grouping:
            #     label_name = self.grouping[label_name]

            label_id = self.label_to_index(label_name)
            print("first label_id : ",label_id)
            frames = sample["threed_pose"]
            T = len(frames)
            if T == 0:
                continue
            # Frame sampling / padding
            if T >= self.clip_len:
                selected_indices = np.linspace(0, T - 1, self.clip_len).astype(int)
            else:
                selected_indices = list(range(T)) + [T - 1] * (self.clip_len - T)

            pose3d_clip = []
            for i in selected_indices:
                #print(i, T)
                frame = frames[i]["keypoints"]
                kp_3d = np.array([
                    [frame[str(j)]["x"], frame[str(j)]["y"], frame[str(j)]["z"]]
                    for j in range(17)
                ])
                pose3d_clip.append(kp_3d)

            pose3d_clip = np.stack(pose3d_clip)  # (T, 17, 3)
            #dummy_person = np.zeros_like(pose3d_clip)
            #combined = np.stack([pose3d_clip, dummy_person], axis=0)  # (2, T, 17, 3)
            print("second label_id : ",label_id)
            all_data.append((pose3d_clip, label_id))

        print(f"Loaded {len(all_data)} samples for split ")
        return all_data

    def label_to_index(self, label):
        label_map = {
            # takeoff / entry
            "takeoff": "takeoff",

            # turning maneuvers
            "cutback": "turn",
            "roundhouse_cutback": "turn",
            "snap": "turn",
            "basic_turn": "turn",
            "bottom_turn": "turn",
            "roller": "turn",
            # aerials
            "air_360": "aerial",
            "air_reversed": "aerial",
            "kick": "aerial",

            # nose riding
            "hang_five": "nose_riding",
            "hang_ten": "nose_riding",

            # tricks
            #"tail_slide": "trick",

            # classic maneuvers
            #"roller": "maneuver",
            "floater": "maneuver",
            "barrel": "maneuver",
            "tail_slide": "maneuver",
            # wipeout stays alone
            "wipeout": "wipeout",
        }

        grouped_label = label_map[label]
        #print(grouped_label)
        grouped_index = {
            "takeoff": 0,
            "turn": 1,
            "aerial": 2,
            "nose_riding": 3,
            "maneuver": 4,
            "wipeout": 5,
        }

        #print(grouped_index[grouped_label])

        return grouped_index[grouped_label]

    def index_to_label(self, index):
        index_map = {
            0: "takeoff",
            1: "turn",
            2: "aerial",
            3: "nose_riding",
            4: "maneuver",
            5: "wipeout"
        }
        return index_map[index]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_tensor, label = self.samples[idx]
        return torch.tensor(input_tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
