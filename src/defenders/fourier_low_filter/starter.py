import copy

import torch
import tqdm
from loguru import logger
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .filter import low_filter
from ...utils.distance_utils import eval_cd
from ...utils.model_eval import PerfTrackVal


def start_gsp_low_filter(
    victim_loader: DataLoader,
    device: str,
    model: Module,
    cutoff: float = 0.6695425186640739,
    key_clean_pc: str = "real_data",
    key_adv_pc: str = "attacked_data",
    key_clean_label: str = "real_label",
    key_adv_label: str = "target_label",
):
    logger.info("Starting GSP low pass filter")
    all_defend_pc = []
    for i, data_item in enumerate(tqdm(victim_loader)):
        batched_adv_pc = data_item[key_adv_pc].to(device).float()
        batched_clean_pc = data_item[key_clean_pc].to(device).float()
        adv_label = torch.squeeze(data_item[key_adv_label]).to(device)
        clean_label = torch.squeeze(data_item[key_clean_label]).to(device)

        victim_pc = copy.deepcopy(data_item["attacked_data"]).float().to(device)
        test_label = data_item["real_label"]
        filter_res = low_filter(victim_pc, cutoff=cutoff)
        clean_input_item = dict()
        clean_input_item["pc"] = filter_res
        clean_input_item["label"] = test_label.detach().clone()

        eval_cd(batched_clean_pc, batched_adv_pc, clean_input_item["pc"])

        all_defend_pc.append(clean_input_item)

    perf = PerfTrackVal()

    for i, data_item in enumerate(tqdm(all_defend_pc, total=len(all_defend_pc))):
        pc = data_item["pc"]
        if isinstance(pc, dict):
            pc = pc["pc"]
        pc = pc.to(device)
        perf.update(model.forward(pc)["logit"].detach().clone(), data_item["label"])

    res = perf.agg()

    logger.info("defenced acc:")
    logger.info(res)

    return res
