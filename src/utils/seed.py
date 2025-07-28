import dgl
import torch_geometric


def seed_everything(seed):
    torch_geometric.seed_everything(seed)
    dgl.seed(seed)
