import copy

import numpy as np
import torch
import tqdm
from loguru import logger
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .SRS import SRSDefense
from ...utils.model_eval import PerfTrackVal


def start_srs(
    victim_loader: DataLoader, device: str, task: str, model: Module, dataset_name: str
):
    defense_module = SRSDefense(drop_num=500)
    all_defend_pc = []

    for i, data_batch in enumerate(tqdm(victim_loader)):
        victim_pc = copy.deepcopy(data_batch["attacked_data"]).float().to(device)
        test_label = data_batch["real_label"]

        srs_res = defense_module.forward(victim_pc)
        clean_input_item = dict()
        clean_input_item["pc"] = [
            item.detach().cpu().numpy().astype(np.float32) for item in srs_res
        ]

        clean_input_item["label"] = test_label.detach().clone()
        clean_input_item["pc"] = torch.tensor(clean_input_item["pc"], device=device)

        all_defend_pc.append(clean_input_item)

    perf = PerfTrackVal()

    for i, data_batch in enumerate(tqdm(all_defend_pc, total=len(all_defend_pc))):
        pc = data_batch["pc"]
        if isinstance(pc, dict):
            pc = pc["pc"]
        pc = pc.to(device)
        perf.update(model.forward(pc)["logit"].detach().clone(), data_batch["label"])

    res = perf.agg()

    logger.info("defenced acc:")
    logger.info(res)

    return res
