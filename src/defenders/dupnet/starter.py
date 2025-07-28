import copy

import torch
import tqdm
from loguru import logger
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .DUP_Net import DUPNet
from src.utils.model_eval import PerfTrackVal


def start_dup(
        victim_loader: DataLoader,
        device: str,
        task: str,
        model: Module,
        dataset_name: str,
        sor_k: int,
        sor_alpha: float):
    logger.info("Starting DUP")
    defense_module = DUPNet(
        sor_k=sor_k,
        sor_alpha=sor_alpha,
        npoint=1024,
        up_ratio=4,
        use_sor=True,
    )
    defense_module.pu_net.load_state_dict(torch.load('./ckpt/dup_net/pu-in_1024-up_4.pth'))
    defense_module.pu_net = defense_module.pu_net.to(device)

    all_defend_pc = []

    for i, data_batch in enumerate(tqdm(victim_loader)):
        victim_pc = copy.deepcopy(data_batch["pc"]).float().to(device)
        test_label = data_batch["label"]
        dup_res = defense_module.forward(victim_pc)

        clean_input_item = dict()
        clean_input_item['pc'] = dup_res
        clean_input_item['label'] = test_label.detach().clone()

        all_defend_pc.append(clean_input_item)

    perf = PerfTrackVal()

    for i, data_batch in enumerate(tqdm(all_defend_pc, total=len(all_defend_pc))):
        pc = data_batch["pc"]
        pc = pc.to(device)
        pred = model.forward(pc)
        perf.update(data_batch=data_batch, out=pred)

    res = perf.agg()

    logger.info("defenced acc:")
    logger.info(res)

    return res
