from train_action import *
from make_windows_from_vid import *
from tqdm import tqdm 

def inference(pose_seq_3d, model, device):
    """
    pose_seq_3d: [N, F, 17, D]
    """
    
    pose_seq_3d = torch.from_numpy(pose_seq_3d.astype('float32'))
    pose_seq_3d = pose_seq_3d.to(device)
    #print(f"pose_seq_3d shape in inference: {pose_seq_3d.shape}")
    with torch.no_grad():
        output = model(pose_seq_3d)
    return output
    
def label_to_index(self, label):
        label_map = {
            "360": 0,
            "cutback-frontside": 1,
            "roller": 2,
            "take-off": 3
        }
        return label_map[label]

def index_to_label(index):
    label_map = {
        0: "360",
        1: "cutback-frontside",
        2: "roller",
        3: "take-off"
    }
    return label_map.get(index, "idle")


if __name__ == "__main__":
    opts = parse_args()
    print(opts)
    #args = get_config('/home/cesar/Desktop/code/models_trial/MotionBERT_custom/configs/action/MB_train_NTU60_xview.yaml')
    parser = argparse.ArgumentParser(description="Inference script for MotionBERT")
    parser.add_argument("--seq_3d_path", type=str, required=True, help="Path to the 3D pose sequence JSON file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration YAML file")
    opts = parser.parse_args()
    seq_3d_path = opts.seq_3d_path
    config_path = opts.config_path
    args = get_config(config_path)
    model_backbone = load_backbone(args)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    criterion = criterion.to(device)
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    pose_seq_3d = make_seq(seq_3d_path)
    # Normalize each frame by its maximum value
    print(f"pose_seq_3d shape: {pose_seq_3d.shape}")
    
    parser.add_argument("--window_size", type=int, default=20, help="Size of the sliding window")
    parser.add_argument("--stride", type=int, default=1, help="Stride for the sliding window")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for classification confidence")
    opts = parser.parse_args()
    
    window_size = opts.window_size
    stride = opts.stride
    threshold = opts.threshold

    windows = sliding_window_pose_sequences(pose_seq_3d, window_size, stride)
    #labels = np.zeros(len(windows))
    labels = []
    #outputs = inference(windows[0], model, device)

    print(windows.shape)
    pose_full = np.expand_dims(pose_seq_3d,axis=0)
    print("pose_full shape:", pose_full.shape)
    output = inference(pose_full, model, device)
    probs = torch.nn.functional.softmax(output, dim=1)
    print("probs : ",probs)
    print("max prob: ", float(torch.max(probs)))
    
    label = int(torch.argmax(probs))
    
     
    print(f"Label for the entire sequence: {index_to_label(label)}")

    for i,window in enumerate(windows):
        window = np.expand_dims(window, axis=0)
        output = inference(window, model, device)
        probs = torch.nn.functional.softmax(output, dim=1)
        if torch.max(probs) > threshold:
            label = int(torch.argmax(probs))
        else:
            label = -1
        labels.append(index_to_label(label))
    print(labels)   

    