from src.utils.model_eval import get_num_classes_by_dataset_name
from .backbone import *


class PointMLP(nn.Module):
    def __init__(
        self,
        dataset: str,
        points=1024,
        class_num=40,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=False,
        use_xyz=False,
        normalize="center",
        dim_expansion=[2, 2, 2, 2],
        pre_blocks=[2, 2, 2, 2],
        pos_blocks=[2, 2, 2, 2],
        k_neighbors=[32, 32, 32, 32],
        reducers=[2, 2, 2, 2],
        **kwargs,
    ):
        super(PointMLP, self).__init__()
        class_num = get_num_classes_by_dataset_name(dataset)
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert (
            len(pre_blocks)
            == len(k_neighbors)
            == len(reducers)
            == len(pos_blocks)
            == len(dim_expansion)
        ), "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(
                last_channel, anchor_points, kneighbor, use_xyz, normalize
            )  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(
                last_channel,
                out_channel,
                pre_block_num,
                groups=groups,
                res_expansion=res_expansion,
                bias=bias,
                activation=activation,
                use_xyz=use_xyz,
            )
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(
                out_channel,
                pos_block_num,
                groups=groups,
                res_expansion=res_expansion,
                bias=bias,
                activation=activation,
            )
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num),
        )

    def forward(self, pc, **kwags):
        if pc.shape[-1] == 3:
            pc = pc.permute(0, 2, 1)

        xyz = pc.permute(0, 2, 1)
        batch_size, _, _ = pc.size()
        pc = self.embedding(pc)  # B,D,N
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, pc = self.local_grouper_list[i](
                xyz, pc.permute(0, 2, 1)
            )  # [b,g,3]  [b,g,k,d]
            pc = self.pre_blocks_list[i](pc)  # [b,d,g]
            pc = self.pos_blocks_list[i](pc)  # [b,d,g]

        pc = F.adaptive_max_pool1d(pc, 1).squeeze(dim=-1)
        pc = self.classifier(pc)
        out = {"logit": pc}
        return out


def PointMLPElite(dataset: str, **kwargs) -> PointMLP:
    return PointMLP(
        dataset=dataset,
        points=1024,
        class_num=get_num_classes_by_dataset_name(dataset),
        embed_dim=32,
        groups=1,
        res_expansion=0.25,
        activation="relu",
        bias=False,
        use_xyz=False,
        normalize="anchor",
        dim_expansion=[2, 2, 2, 1],
        pre_blocks=[1, 1, 2, 1],
        pos_blocks=[1, 1, 2, 1],
        k_neighbors=[24, 24, 24, 24],
        reducers=[2, 2, 2, 2],
        **kwargs,
    )
