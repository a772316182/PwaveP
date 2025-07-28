import torch.nn as nn

from src.utils.model_eval import get_num_classes_by_dataset_name
from .backbone import DGCNN as Backbone


class DGCNN(nn.Module):

    def __init__(self, dataset: str):
        super().__init__()
        self.dataset = dataset
        num_classes = get_num_classes_by_dataset_name(dataset)

        self.model = Backbone(
            k=20,
            leaky_relu=True,
            emb_dims=1024,
            dropout=0.5,
            output_channels=num_classes,
        )

    def forward(self, pc, **kwargs):
        if pc.shape[-1] != 3:
            pc = pc.permute(0, 2, 1)

        pc = pc.float()
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        logit = self.model(pc)
        return {"logit": logit}
