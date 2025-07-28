import torch.nn as nn

from src.utils.model_eval import get_num_classes_by_dataset_name
from .backbone import PointNet as Backbone


class PointNet(nn.Module):

    def __init__(self, dataset: str):
        super().__init__()
        num_class = get_num_classes_by_dataset_name(dataset)
        self.model = Backbone(k=num_class, feature_transform=True)

    def forward(self, pc, **kwags):
        if pc.shape[-1] != 3:
            pc = pc.permute(0, 2, 1)

        pc = pc.transpose(2, 1).float()
        logit, _, trans_feat = self.model(pc)
        out = {"logit": logit, "trans_feat": trans_feat}
        return out
