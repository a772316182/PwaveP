import os
from typing import Tuple

import torch
import tqdm
from loguru import logger
from tqdm import tqdm

from src.datasets.common_loader import create_clean_data_loader
from src.models.curvenet.index import CurveNet
from src.models.dgcnn.index import DGCNN
from src.models.evaluator import validate
from src.models.pct.index import Pct
from src.models.pointmlp.index import PointMLP, PointMLPElite
from src.models.pointnet.index import PointNet
from src.models.pointnet2.index import PointNet2ClsMsg
from src.utils.save_utils import pre_check_path


def init_model(model_name, dataset_name):
    if model_name == "dgcnn":
        model = DGCNN(dataset=dataset_name)
    elif model_name == "pointnet":
        model = PointNet(dataset=dataset_name)
    elif model_name == "pct":
        model = Pct(dataset=dataset_name)
    elif model_name == "pointnet2":
        model = PointNet2ClsMsg(dataset=dataset_name)
    elif model_name == "curvenet":
        model = CurveNet(dataset=dataset_name)
    elif model_name == "pointmlp":
        model = PointMLP(dataset=dataset_name)
    elif model_name == "pointmlpelite":
        model = PointMLPElite(dataset=dataset_name)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    return model


def get_checkpoint_path(model_name, dataset_name, ckpt_root) -> Tuple[str, bool]:
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    path = os.path.join(ckpt_root, f"{dataset_name}_{model_name}.pth")
    return path, os.path.exists(path)


def load_model(
        device,
        model_name,
        dataset_name,
        ckpt_root,
):
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    path, exists = get_checkpoint_path(model_name, dataset_name, ckpt_root)

    assert exists, f"Checkpoint {path} does not exist, please check"

    model = init_model(model_name, dataset_name).to(device=device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    logger.info('No need to train...')
    logger.info(f"Loaded model from {path}")
    model.eval()
    return model


def load_dataset(batch_size, dataset_name, data_root):
    dataset_name = dataset_name.lower()
    loader_test = create_clean_data_loader(
        split="test",
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=4,
        data_root=data_root,
    )
    loader_train = create_clean_data_loader(
        split="train",
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=4,
        data_root=data_root,
    )

    return loader_train, loader_test


def train_model(
        device,
        model_name,
        dataset_name,
        data_root,
        ckpt_root,
        batch_size,
        num_epochs,
        lr,
        weight_decay,
        scheduler_step,
        scheduler_gamma,
):
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    path, exists = get_checkpoint_path(model_name, dataset_name, ckpt_root)
    model = init_model(model_name, dataset_name).to(device=device)

    logger.info(f"state dict on {model_name} @ {dataset_name}")
    i = 0
    for k, v in model.state_dict().items():
        i += 1
        logger.info(f'{i} \t {k}')

    loader_test = create_clean_data_loader(
        split="test",
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=4,
        data_root=data_root,
    )

    if exists:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info('No need to train...')
        logger.info(f"Loaded model from {path}")
    else:
        logger.info(f"{path} does not exist, start training...")
        loader_train = create_clean_data_loader(
            split="train",
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=4,
            data_root=data_root,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )

        for epoch in tqdm(range(num_epochs), desc="Epoch"):
            for batch_idx, data in enumerate(tqdm(loader_train, desc="Batch", leave=False)):
                model.train()
                optimizer.zero_grad()
                pc, label = data["pc"], data["label"]
                pc, label = pc.to(device=device), label.to(device=device)
                pred = model.forward(pc)["logit"]
                loss = torch.nn.CrossEntropyLoss()(pred, label)
                loss.backward()
                optimizer.step()
                scheduler.step()
            if epoch % 1 == 0:
                model.eval()
                res = validate(
                    loader=loader_test,
                    model=model,
                    device=device,
                )
                logger.info(res)

    model.eval()
    res = validate(
        loader=loader_test,
        model=model,
        device=device,
    )
    logger.info(res)
    torch.save(model.state_dict(), pre_check_path(path))
    return model
