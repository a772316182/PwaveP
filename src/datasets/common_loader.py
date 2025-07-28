import os

import torch
from torch.utils.data import DataLoader

from .modelnet40 import ModelNet40
from .shapenet import ShapeNetPart


def create_clean_data_loader(dataset_name, data_root, split, batch_size, num_workers):
    train_data_path, valid_data_path, test_data_path = _clean_data_location(dataset_name, data_root)
    return _fetch_clean_data(
        split=split,
        dataset_name=dataset_name,
        train_data_path=train_data_path,
        valid_data_path=valid_data_path,
        test_data_path=test_data_path,
        num_points=1024,
        batch_size=batch_size,
        num_workers=num_workers,
        data_root=data_root
    )


def _clean_data_location(dataset_name: str, data_root: str):
    if dataset_name == "modelnet40":
        train_data_path = f"{data_root}/modelnet40_ply_hdf5_2048/train_files.txt"
        valid_data_path = f"{data_root}/modelnet40_ply_hdf5_2048/train_files.txt"
        test_data_path = f"{data_root}/modelnet40_ply_hdf5_2048/test_files.txt"
    elif dataset_name == "shapenetpart":
        train_data_path = f"{data_root}/shapenetpart_hdf5_2048/train_files.txt"
        valid_data_path = f"{data_root}/shapenetpart_hdf5_2048/train_files.txt"
        test_data_path = f"{data_root}/shapenetpart_hdf5_2048/test_files.txt"
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")
    src = os.getcwd()
    train_data_path = os.path.join(src, train_data_path)
    valid_data_path = os.path.join(src, valid_data_path)
    test_data_path = os.path.join(src, test_data_path)
    return train_data_path, valid_data_path, test_data_path


def _fetch_clean_data(
        split,
        dataset_name,
        data_root,
        train_data_path,
        valid_data_path,
        test_data_path,
        num_points,
        batch_size,
        num_workers,
):
    if dataset_name == "modelnet40":
        dataset = ModelNet40(
            train_data_path=train_data_path,
            valid_data_path=valid_data_path,
            test_data_path=test_data_path,
            num_points=num_points,
            split=split,
            data_root=data_root,
        )
    elif dataset_name == "shapenetpart":
        dataset = ShapeNetPart(
            train_data_path=train_data_path,
            valid_data_path=valid_data_path,
            test_data_path=test_data_path,
            num_points=num_points,
            split=split,
            data_root=data_root,
        )
    else:
        raise NotImplementedError(
            f"Unsupported dataset: {dataset_name}, available: modelnet40, shapenetpart"
        )

    if "batch_proc" not in dir(dataset):
        dataset.batch_proc = None

    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        drop_last=True,
        pin_memory=(torch.cuda.is_available()) and (not num_workers),
    )
