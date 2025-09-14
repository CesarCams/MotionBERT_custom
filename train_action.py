import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import random
import tensorboardX
from tqdm import tqdm
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from lib.data.dataset_surf import SurfActionDataset, SurfActionDatasetV2
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, random_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import torch
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import json
from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import ActionNet
from sklearn.preprocessing import StandardScaler


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-dr','--data_root', default='lib/data/processed_videos', type=str, metavar='PATH', help='root directory of the dataset')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('--clip_len', default=50, type=int, help='Number of frames per clip')
    opts = parser.parse_args()
    return opts

def flatten_pose(sample_input):
        # sample_input: (2, T, 17, 4)
        first_person = sample_input[0]  # (T, 17, 4)

        # Optionally downsample temporally (e.g., every 4th frame)
        #first_person = first_person[::4]  # shape: (T/4, 17, 4)

        # Flatten
        return first_person.flatten()  # shape: (T/4 * 17 * 4,)

def validate(test_loader, model, criterion,device):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size = len(batch_input)    
            batch_gt = batch_gt.to(device)
            batch_input = batch_input.to(device)
            output = model(batch_input)    # (N, num_classes)
            loss = criterion(output, batch_gt)

            # update metric
            losses.update(loss.item(), batch_size)
            topk = (1, min(5, args.action_classes))
            acc1, acc5 = accuracy(output, batch_gt, topk=topk)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % opts.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = torch.nn.CrossEntropyLoss()
    # if torch.backends.mps.is_available():
    #     device = torch.device("cpu")
    #     print("Using CPU anyway")
    # else:
    #     device = torch.device("cpu")
    #     print("Using CPU")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    criterion = criterion.to(device)

    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 2,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    testloader_params = {
          'batch_size': args.batch_size, 
          'shuffle': False,
          'num_workers': 2,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    
    # with open("lib/data/splits/train_list_0.5.txt") as f:
    #     list_unshuffled = [line.strip() for line in f]
    #     train_list = random.sample(list_unshuffled,k=len(list_unshuffled))

    # with open("lib/data/splits/test_list_0.5.txt") as f:
    #     list_unshuffled = [line.strip() for line in f]
    #     full_test_list = random.sample(list_unshuffled,k=len(list_unshuffled))

    # # Split test_list into validation and test (50/50)
    # split_index = len(full_test_list) // 2
    # val_list = full_test_list[:split_index]
    # test_list = full_test_list[split_index:]

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

    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    
    if not opts.evaluate:
        # Unwrap DataParallel if necessary
        backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
        head     = model.module.head     if isinstance(model, nn.DataParallel) else model.head

        optimizer = optim.AdamW(
            [
                {"params": filter(lambda p: p.requires_grad, backbone.parameters()), "lr": args.lr_backbone},
                {"params": filter(lambda p: p.requires_grad, head.parameters()),     "lr": args.lr_head},
            ],
            weight_decay=args.weight_decay
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        print('INFO: Training on {} batches'.format(len(train_loader)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            end = time.time()
            iters = len(train_loader)
            for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):    # (N, 2, T, 17, 3)
                #print(idx)
                data_time.update(time.time() - end)
                batch_size = len(batch_input)
                batch_gt = batch_gt.to(device)
                batch_input = batch_input.to(device)
                output = model(batch_input) # (N, num_classes)
                optimizer.zero_grad()
                loss_train = criterion(output, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                topk = (1, min(5, args.action_classes))
                acc1, acc5 = accuracy(output, batch_gt, topk=topk)
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                loss_train.backward()
                optimizer.step()    
                batch_time.update(time.time() - end)
                end = time.time()
            if (idx + 1) % opts.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses_train, top1=top1))
                sys.stdout.flush()
                
            test_loss, test_top1, test_top5 = validate(val_loader, model, criterion,device)
                
            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_top1', top1.avg, epoch + 1)
            train_writer.add_scalar('train_top5', top5.avg, epoch + 1)
            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            train_writer.add_scalar('test_top1', test_top1, epoch + 1)
            train_writer.add_scalar('test_top5', test_top5, epoch + 1)
            
            scheduler.step()

            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
            }, chk_path)

            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            if test_top1 > best_acc:
                best_acc = test_top1
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)

    if opts.evaluate:
        print("Perf on training set : ")
        evaluate_model(model,train_loader,device,name='conf_matrix_train.png')
        print("Perf on test set : ")
        evaluate_model(model,test_loader,device,name='conf_matrix_test.png')

        # print("Evaluating model...")
        # test_loss, test_top1, test_top5 = validate(test_loader, model, criterion,device)
        # print('Loss {loss:.4f} \t'
        #       'Acc@1 {top1:.3f} \t'
        #       'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))
    
def train_basic_class():
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
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'),
            "params": {
                "C": [0.01, 0.1, 1, 10]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        },
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

        with open("best_params.json", "w") as f:
            json.dump({
                "model": best_name,
                "params": best_params,
                "accuracy": best_score
            }, f, indent=4)

        print("✅ Model saved to best_model.pkl")
        print("✅ Parameters saved to best_params.json")
        
    cm = confusion_matrix(y_test, y_pred_best)
    labels = sorted(np.unique(y_test))  # class indices, e.g., [0, 1, 2]

    # If you want class names (optional)
    class_names = [f"Class {i}" for i in labels]

    # --- Option 1: Pretty Confusion Matrix with seaborn ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device, class_names=None,name='conf_matrix.png'):
    model.eval()
    all_preds = []
    all_labels = []
    if class_names is None:
        class_names = [0,1,2,3]
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Unpack batch (assuming (input, label) structure)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            print(f"Input shape: {inputs.shape}, Labels shape: {labels.shape}")
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Test Accuracy: {acc * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(name)
    #plt.show()

    return acc, cm


if __name__ == "__main__":
    args = parse_args()
    #print(opts)
    #args = get_config(opts.config)

    # train_with_config(args, opts)
    train_basic_class()