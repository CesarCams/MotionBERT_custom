from train_action import *
from make_windows_from_vid import *
from tqdm import tqdm 
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
import json
from scipy.special import softmax

def index_to_label(index):
        index_map = {
            0: "takeoff",
            1: "turn",
            2: "aerial",
            3: "nose_riding",
            4: "maneuver",
            5: "wipeout",
            -1: "idle"
        }
        return index_map[index]

def inference(pose_seq_3d, model, device):
    """
    pose_seq_3d: [N, F, 17, D]
    """
    
    #pose_seq_3d = torch.from_numpy(pose_seq_3d.astype('float32'))
    #pose_seq_3d = pose_seq_3d.to(device)
    #print(f"pose_seq_3d shape in inference: {pose_seq_3d.shape}")
    #with torch.no_grad():
    output = model.predict_proba(pose_seq_3d)
    return output
    
def build_model_test():
    trainloader_params = {
        'batch_size': 8,   # ← bigger batch size for faster preprocessing
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }
    testloader_params = {
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    with open(args.data_root, "r") as f:
        dataset = json.load(f)

    num_samples = len(dataset["samples"])
    indices = list(range(num_samples))

    # Simple 50/50 split (change if needed)
    train_list = indices[: int(0.5 * num_samples)]
    test_list = indices[int(0.5 * num_samples):]

    # Create datasets
    train_dataset = SurfActionDatasetV2(json_path=args.data_root, split_list=train_list, clip_len=args.clip_len)
    test_dataset = SurfActionDatasetV2(json_path=args.data_root, split_list=test_list, clip_len=args.clip_len)

    # Create loaders
    train_loader = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)

    X_train, y_train = [], []
    X_test, y_test = [], []

    # --- Collect training data ---
    for sample_input, labels in tqdm(train_loader):
        # sample_input: (batch, T, 17, 3)
        # Flatten the whole batch at once
        flat_batch = sample_input.view(sample_input.size(0), -1).numpy()
        X_train.append(flat_batch)
        y_train.extend(labels.numpy())

    # --- Collect testing data ---
    for sample_input, labels in tqdm(test_loader):
        flat_batch = sample_input.view(sample_input.size(0), -1).numpy()
        X_test.append(flat_batch)
        y_test.extend(labels.numpy())

    # Concatenate all batches
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Shape of training set:", X_train.shape)
    print("Shape of test set:", X_test.shape)
    print("Labels in training set:", set(y_train))
    print("Labels in test set:", set(y_test))

    # --- Scale features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Define models ---
    models_and_grids = {
        "MLP": {
            "model": MLPClassifier(max_iter=500, random_state=42),  # reduce max_iter for speed
            "params": {
                "hidden_layer_sizes": [(100,), (100, 50)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001]
            }
        }
    }

    best_model, best_score, best_name, best_params = None, 0.0, "", {}
    results = {}

    for name, entry in models_and_grids.items():
        print(f"\n🔍 Training {name}...")
        gs = GridSearchCV(entry["model"], entry["params"], cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        gs.fit(X_train_scaled, y_train)

        y_pred = gs.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"✅ {name} Validation Accuracy: {acc:.4f}")
        print("📊 Best Parameters:", gs.best_params_)
        print(classification_report(y_test, y_pred))

        if acc > best_score:
            best_score = acc
            best_model = gs.best_estimator_
            best_params = gs.best_params_
            best_name = name

    print(f"\n🏆 Best Model: {best_name} with Accuracy: {best_score:.4f}")

    with open("best_params.json", "w") as f:
        json.dump({"model": best_name, "params": best_params, "accuracy": best_score}, f, indent=4)

    return best_model, scaler


def build_model():
    trainloader_params = {
            'batch_size': 8,
            'shuffle': True,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True
    }
    testloader_params = {
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    with open(args.data_root, "r") as f:
        dataset = json.load(f)

    num_samples = len(dataset["samples"])
    indices = list(range(num_samples))

    # Example simple split (80/10/10)
    train_list = indices[: int(0.5 * num_samples)]
    #val_list = indices[int(0.8 * num_samples): int(0.9 * num_samples)]
    test_list = indices[int(0.5 * num_samples):]

    # Create datasets

    # Create datasets
    train_dataset = SurfActionDatasetV2(json_path=args.data_root, split_list=train_list, clip_len=args.clip_len)
    #val_dataset = SurfActionDatasetV2(json_path=args.data_root, split_list=val_list, clip_len=args.clip_len)
    test_dataset = SurfActionDatasetV2(json_path=args.data_root, split_list=test_list, clip_len=args.clip_len)

    # Create loaders
    train_loader = DataLoader(train_dataset, **trainloader_params)
    #val_loader = DataLoader(val_dataset, **testloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)

    X_train, y_train = [], []
    X_test, y_test = [], []
    for sample_input, label in tqdm(train_loader):
        #embed = model.extract_embedding(torch.tensor(sample_input.numpy()))
        #print(embed.shape)
        for i in range(sample_input.size(0)):
            #X_train.append(embed[i].detach().numpy())
            X_train.append(flatten_pose(sample_input[i].numpy()))
            #print(X_train[0].shape)

            y_train.append(label[i].item())

    print("Shape of first training sample:", X_train[0].shape)
    for sample_input, label in tqdm(test_loader):
        #for i in range(sample_input.size(0)):
            #if label[i].item()==3:
            #    continue
        #embed = model.extract_embedding(torch.tensor(sample_input.numpy()))

        for i in range(sample_input.size(0)):
            #X_test.append(embed[i].detach().numpy())
            X_test.append(flatten_pose(sample_input[i].numpy()))
            y_test.append(label[i].item())
            
            
            #y_pred.append()
    print("Labels in training set:", set(y_train))
    print("Labels in test set:", set(y_test))

    X_train = np.stack(X_train)
    X_test = np.stack(X_test)


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models_and_grids = {
        # "LogisticRegression": {
        #     "model": LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'),
        #     "params": {
        #         "C": [0.01, 0.1, 1, 10]
        #     }
        # },
        # "RandomForest": {
        #     "model": RandomForestClassifier(random_state=42),
        #     "params": {
        #         "n_estimators": [100, 200],
        #         "max_depth": [10, None],
        #         "min_samples_split": [2, 5],
        #         "min_samples_leaf": [1, 2]
        #     }
        # },
        # "SVM": {
        #     "model": SVC(probability=True),
        #     "params": {
        #         "C": [0.1, 1, 10],
        #         "kernel": ["linear", "rbf"]
        #     }
        #},
        "MLP": {
            "model": MLPClassifier(max_iter=1000, random_state=42),
            "params": {
                "hidden_layer_sizes": [(100,), (100, 50)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001]
            }
        }
    }

    best_model = None
    best_score = 0.0
    best_name = ""
    best_params = {}
    results = {}

    # --- Run Grid Search for each model ---
    for name, entry in models_and_grids.items():
        print(f"\n🔍 Training {name}...")
        clf = entry["model"]
        grid = entry["params"]

        gs = GridSearchCV(clf, grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        gs.fit(X_train_scaled, y_train)

        y_pred = gs.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"✅ {name} Validation Accuracy: {acc:.4f}")
        print("📊 Best Parameters:", gs.best_params_)
        print(classification_report(y_test, y_pred))

        if acc > best_score:
            best_score = acc
            y_pred_best = y_pred
            best_model = gs.best_estimator_
            best_params = gs.best_params_
            best_name = name

    # --- Save best model and parameters ---
    print(f"\n🏆 Best Model: {best_name} with Accuracy: {best_score:.4f}")
    #joblib.dump(best_model, "best_model.pkl")

    # with open("best_params.json", "w") as f:
    #     json.dump({
    #         "model": best_name,
    #         "params": best_params,
    #         "accuracy": best_score
    #     }, f, indent=4)

    print("✅ Model saved to best_model.pkl")
    print("✅ Parameters saved to best_params.json")
    return best_model, scaler
    
if __name__ == "__main__":
    #args = get_config('/home/cesar/Desktop/code/models_trial/MotionBERT_custom/configs/action/MB_train_NTU60_xview.yaml')
    parser = argparse.ArgumentParser(description="Inference script for MotionBERT")
    parser.add_argument("--seq_3d_path", type=str, required=False,default="/Users/cesarcamusemschwiller/Desktop/Pro/Surfeye/code/maneuver-recognition/video-labeler/metadata/ab_right_camera_2025-8-31/2025-08-31-11-01-46_1213.json",  help="Path to the 3D pose sequence JSON file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration YAML file")
    parser.add_argument("--window_size", type=int, default=50, help="Size of the sliding window")
    parser.add_argument("--stride", type=int, default=10, help="Stride for the sliding window")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for classification confidence")
    opts = parser.parse_args()
    seq_3d_path = opts.seq_3d_path
    config_path = opts.config_path
    args = get_config(config_path)
    #model_backbone = load_backbone(args)
    #model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    # Load the best model from the pickle file


    model, scaler = build_model()
    print("Model built and trained.")
    # criterion = torch.nn.CrossEntropyLoss()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")
    #model = model.to(device)
    #criterion = criterion.to(device)
    # if opts.resume or opts.evaluate:
    #     chk_filename = opts.evaluate if opts.evaluate else opts.resume
    #     print('Loading checkpoint', chk_filename)
    #     checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    #     model.load_state_dict(checkpoint['model'], strict=True)
    #model.eval()

    pose_seq_3d = make_seq(seq_3d_path)
    # Normalize each frame by its maximum value
    print(f"pose_seq_3d shape: {pose_seq_3d.shape}")
    
    
    #opts = parser.parse_args()
    
    window_size = opts.window_size
    stride = opts.stride
    threshold = opts.threshold

    windows = sliding_window_pose_sequences(pose_seq_3d, window_size, stride)

    labels = []

    print(windows.shape)

    for i, window in enumerate(windows):
        # flatten + scale
        scaled_window = scaler.transform([flatten_pose(window)])

        # inference (model is sklearn or numpy-based)
        output = inference(scaled_window, model, None)   # (1, num_classes)

        # probabilities
        probs = softmax(output, axis=1)  # numpy array

        # confidence check
        if np.max(probs) > threshold:
            label = int(np.argmax(probs))
        else:
            label = -1

        frame_start = i * stride
        frame_end = frame_start + window_size - 1
        print(f"Window {i} (frames {frame_start}–{frame_end}) → Label: {index_to_label(label)}, Probs: {probs}")

        labels.append(index_to_label(label))

    print(labels)


        