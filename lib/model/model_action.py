import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActionHeadClassification(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_classes=60, num_joints=17, hidden_dim=2048):
        super(ActionHeadClassification, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)    
        feat = self.fc2(feat)
        return feat
        
class ActionHeadEmbed(nn.Module):
    def __init__(self, dropout_ratio=0., dim_rep=512, num_joints=17, hidden_dim=2048):
        super(ActionHeadEmbed, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(dim_rep*num_joints, hidden_dim)
    def forward(self, feat):
        '''
            Input: (N, M, T, J, C)
        '''
        N, M, T, J, C = feat.shape
        feat = self.dropout(feat)
        feat = feat.permute(0, 1, 3, 4, 2)      # (N, M, T, J, C) -> (N, M, J, C, T)
        feat = feat.mean(dim=-1)
        feat = feat.reshape(N, M, -1)           # (N, M, J*C)
        feat = feat.mean(dim=1)
        feat = self.fc1(feat)
        feat = F.normalize(feat, dim=-1)
        return feat

class ActionNet(nn.Module):
    def __init__(self, backbone, dim_rep=512, num_classes=60, dropout_ratio=0., version='class', hidden_dim=2048, num_joints=17,clip_len=50):
        super(ActionNet, self).__init__()
        self.backbone = backbone
        self.clip_len = clip_len
        self.feat_J = num_joints
        if version=='class':
            self.head = ActionHeadClassification(dropout_ratio=dropout_ratio, dim_rep=dim_rep, num_classes=num_classes, num_joints=num_joints)
        elif version=='embed':
            self.head = ActionHeadEmbed(dropout_ratio=dropout_ratio, dim_rep=dim_rep, hidden_dim=hidden_dim, num_joints=num_joints)
        else:
            raise Exception('Version Error.')
        
    def forward(self, x):
        '''
            Input: (N, M x T x 17 x 3) 
        '''
        N, M, T, J, C = x.shape

        if T >= self.clip_len:
            selected_indices = np.linspace(0, T - 1, self.clip_len).astype(int)
        else:
            # Use all available frames and repeat last frame as needed
            selected_indices = list(range(T)) + [T - 1] * (self.clip_len - T)

        #print("Shaoe before selection:", x.shape)

        x = x[:, :, selected_indices, :, :]  # Select frames based on the indices
        #print("Shape after selection:", x.shape)
        N, M, T, J, C = x.shape
        x = x.reshape(N*M, T, J, C)        
        feat = self.backbone.get_representation(x)
        feat = feat.reshape([N, M, T, self.feat_J, -1])      # (N, M, T, J, C)
        out = self.head(feat)
        return out


    def extract_embedding(self, x):
        N, M, T, J, C = x.shape
        x = x.reshape(N * M, T, J, C)                   # (N*M, T, J, C)
        feat = self.backbone.get_representation(x)     # (N*M, T, J, C_rep)
        feat = feat.reshape(N, M, T, self.feat_J, -1)   # (N, M, T, J, C_rep)
        embedding = self.head(feat)                     # (N, hidden_dim)
        return embedding