from train_action import *
from make_windows_from_vid import *
from tqdm import tqdm 

def inference(pose_seq_3d, model, device):
    """
    pose_seq_3d: [N, F, 17, D]
    """
    
    pose_seq_3d = torch.from_numpy(pose_seq_3d.astype('float32'))
    pose_seq_3d = pose_seq_3d.to(device)
    with torch.no_grad():
        output = model(pose_seq_3d)
    return output
    

if __name__ == "__main__":
    opts = parse_args()
    print(opts)
    args = get_config('/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionBERT_custom/configs/action/MB_train_NTU60_xview.yaml')
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
    seq_3d_path = "/Users/cesarcamusemschwiller/Desktop/Surfeye/code/models_trial/MotionBERT_custom/lib/data/processed_videos/b_2022-11-05-12-20-48_558/keypoints_xyz.json"
    pose_seq_3d = make_seq(seq_3d_path)
    window_size = 50
    stride = 3
    threshold = 0.6
    windows = sliding_window_pose_sequences(pose_seq_3d, window_size, stride)
    #labels = np.zeros(len(windows))
    labels = []
    #outputs = inference(windows[0], model, device)
    print(windows.shape)
    for i,window in enumerate(windows):
        #print(i)
        window = np.expand_dims(window, axis=0)
        print(window.shape)
        output = inference(window, model, device)
        print(output)
        probs = torch.nn.functional.softmax(output, dim=1)
        print(torch.max(probs))
        if torch.max(probs) > threshold:
            label = torch.argmax(probs)
        else:
            label = -1

        print(label)
        labels.append(label)
    #print(labels)   

    