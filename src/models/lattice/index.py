import torch
from torch import nn

from .backbone import resnet50, LatticeGen
from ...utils.model_eval import get_num_classes_by_dataset_name


class Lattice(nn.Module):

    # s is the final image scale
    def __init__(self, dataset: str, s=128 * 3):
        super(Lattice, self).__init__()
        self.dataset = dataset
        num_classes = get_num_classes_by_dataset_name(dataset)

        self.normal_channel = False
        self.lat_transform = LatticeGen(s, normal_channel=self.normal_channel)
        self.size2d = s
        self.network_2d = resnet50(num_cls=num_classes, c=1)

    def forward(self, x):
        if x.shape[-1] != 3:
            x = x.permute(0, 2, 1)

        device = x.device
        if self.normal_channel:
            vv = x[:, 3:]
            # vv = x[:, :3]
        else:
            vv = torch.ones((x.size(0), 1, x.size(2))).to(device)

        # returned splatted has a shape of [b, size, size, c]
        splatted_2d, _, __ = self.lat_transform.forward(
            x[:, :3] * (self.size2d // 2 - 2), vv
        )
        # splatted_2d = torch.cat((splatted_2d, splatted_2d_2), 3).permute(0, 3, 1, 2).contiguous()
        # splatted_2d = splatted_2d.permute(0, 3, 1, 2).contiguous()
        # network takes [b, c, size, size]
        outputs = self.network_2d(splatted_2d)
        return {"logit": outputs}
