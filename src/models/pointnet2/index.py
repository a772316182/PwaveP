import torch.nn as nn
import torch.nn.functional as F

from src.utils.model_eval import get_num_classes_by_dataset_name
from .backbone import PointNetSetAbstraction, PointNetSetAbstractionMsg


class PointNet2ClsSsg(nn.Module):
    def __init__(self, dataset: str):
        super(PointNet2ClsSsg, self).__init__()
        num_classes = get_num_classes_by_dataset_name(dataset)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, pc, **kwags):
        if pc.shape[-1] != 3:
            pc = pc.permute(0, 2, 1)

        B, _, _ = pc.shape
        pc = pc.permute(0, 2, 1)
        l1_xyz, l1_points = self.sa1(pc, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        logit = self.fc3(x)
        out = {"logit": logit}
        return out


class PointNet2ClsMsg(nn.Module):
    def __init__(self, dataset: str):
        super(PointNet2ClsMsg, self).__init__()
        num_classes = get_num_classes_by_dataset_name(dataset)
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 0,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(
            None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, pc, **kwags):
        if pc.shape[-1] != 3:
            pc = pc.permute(0, 2, 1)

        B, _, _ = pc.shape
        pc = pc.permute(0, 2, 1)

        B, _, _ = pc.shape
        l1_xyz, l1_points = self.sa1(pc, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        logit = self.fc3(x)
        out = {"logit": logit}
        return out
