import torch.nn as nn

from .backbone import Pct


class PCT(nn.Module):
    def __init__(self, dataset: str, dropout=0.5):
        super(PCT, self).__init__()
        self.model = Pct(dataset, dropout)

    def forward(self, pc, **kwags):
        if pc.shape[-1] != 3:
            pc = pc.permute(0, 2, 1)

        pc = pc.transpose(2, 1).float()
        return self.model.forward(pc)
