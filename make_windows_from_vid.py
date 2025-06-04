import numpy as np
import json
import pickle
import numpy as np
from torch.utils.data import Dataset


#dir_path = "/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionBERT_custom/lib/data/processed_videos/360/360_3/"
json_path = "/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionBERT_custom/lib/data/processed_videos/b_2022-11-05-12-20-48_558/keypoints_xyz.json"
window_size = 10

def make_seq(json_path):
#pkl_path = f"{dir_path}/input_2D/keypoints.npz"
# Load 3D keypoints from JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    frames = json_data["frames"]

    pose3d_clip = []

    for i in range(len(frames)):
        frame = frames[i]["keypoints"]
        kp_3d = np.array([[frame[str(j)]["x"],
                        frame[str(j)]["y"],
                        frame[str(j)]["z"]] for j in range(17)])  # (17,3)
        max_val = np.max(kp_3d, axis=0, keepdims=True)
        kp_3d = kp_3d / max_val
        pose3d_clip.append(kp_3d)

    pose3d_clip = np.stack(pose3d_clip)  # (T, 17, 3)
    dummy_person = np.zeros_like(pose3d_clip) # (T, 17, 3)
    combined = np.stack([pose3d_clip, dummy_person], axis=0)  # (2, T, 17, 3)
    return combined

def sliding_window_pose_sequences(pose_seq, window_size, stride):
    """
    Breaks 3D pose sequence into sliding windows.

    Args:
        pose_seq: np.array of shape (B, T, 17, 3)
        window_size: int, length of each window
        stride: int, number of frames to move per step

    Returns:
        A list of shape [B][N] where each item is (window_size, 17, 3)
        i.e., output[b][i] = ith window from batch sample b
    """
    B, T, J, C = pose_seq.shape
    windows = []

    
    seq_windows = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        window = pose_seq[:, start:end]  # shape: (window_size, 17, 3)
        seq_windows.append(window)
    windows.append(seq_windows)

    windows = np.array(windows)
    windows = windows[0]
    return windows

window_size = 50
stride = 3
if __name__=="__main__":

    json_path = '/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionAGFormer_custom/samples_json/2025-05-23-17-10-46_17.json'
    combined = make_seq(json_path)

    windows = sliding_window_pose_sequences(combined, window_size, stride=1)
    
    labels = np.zeros(len(windows))

    
    
        
    