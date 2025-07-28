import os.path

import h5py
import numpy as np
from loguru import logger
from torch.utils.data import Dataset


class ModelNet40(Dataset):

    def __init__(
            self, split, data_root, train_data_path, valid_data_path, test_data_path, num_points
    ):
        logger.info("ModelNet40")
        logger.info(f"train_data_path: {train_data_path}")
        logger.info(f"valid_data_path: {valid_data_path}")
        logger.info(f"test_data_path: {test_data_path}")
        logger.info(f"num_points: {num_points}")

        self.data_root = data_root

        self.num_points = num_points
        self.split = split
        self.partition = "train" if split in ["train", "valid"] else "test"

        self.data_path = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test": test_data_path,
        }[self.split]

        self.data_dir = os.path.dirname(train_data_path)

        self.data, self.label = self.load_data(self.data_path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        pc = self.data[idx][: self.num_points]
        label = self.label[idx]
        if self.partition == "train":
            pc = self.translate_pointcloud(pc)
            np.random.shuffle(pc)

        return {"pc": pc, "label": label.item()}

    @staticmethod
    def translate_pointcloud(pointcloud):
        xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
            "float32"
        )
        return translated_pointcloud

    def load_data(self, path):
        all_data = []
        all_label = []
        with open(path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(
                    os.path.join(self.data_dir, h5_name.rstrip())
                    , "r")
                data = f["data"][:].astype("float32")
                label = f["label"][:].astype("int64")
                f.close()
                all_data.append(data)
                all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label
