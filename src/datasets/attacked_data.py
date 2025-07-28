import torch
from torch.utils.data import Dataset


class AttackedData(Dataset):

    def __init__(self, data_dict):
        self.attacked_data = torch.cat([item["attacked_data"] for item in data_dict], dim=0)
        self.real_data = torch.cat([item["real_data"] for item in data_dict], dim=0)
        self.real_label = torch.cat([item["real_label"] for item in data_dict], dim=0)
        self.target_label = torch.cat([item["target_label"] for item in data_dict], dim=0)

    def __len__(self):
        return len(self.attacked_data)

    def __getitem__(self, idx):
        sample = {
            "attacked_data": self.attacked_data[idx],
            "real_data": self.real_data[idx],
            "real_label": self.real_label[idx],
            "target_label": self.target_label[idx],
        }
        return sample
