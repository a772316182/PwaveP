import torch
import tqdm
from loguru import logger
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.model_eval import PerfTrackVal
from .cvar import advanced_cvar


def start_cvar(
        loader: DataLoader, device: str, model: Module,
        key_clean_pc: str = "real_data",
        key_adv_pc: str = "attacked_data",
        key_clean_label: str = "real_label",
        key_adv_label: str = "target_label",
        break_on: int = 10000,
):
    logger.info("Starting CVAR")

    all_defend_pc = []

    for i, data_item in enumerate(tqdm(loader)):
        if i > break_on:
            break
        batched_adv_pc = data_item[key_adv_pc].to(device).float()
        batched_clean_pc = data_item[key_clean_pc].to(device).float()
        adv_label = torch.squeeze(data_item[key_adv_label]).to(device)
        clean_label = torch.squeeze(data_item[key_clean_label]).to(device)

        cvar_res = advanced_cvar(data_batch={
            "pc": batched_adv_pc,
            "label": clean_label
        }, model=model, drop_rate=0.02)

        clean_input_item = dict()
        clean_input_item["cvar_res"] = cvar_res
        clean_input_item["label"] = clean_label.detach().clone()

        all_defend_pc.append(clean_input_item)

    perf = PerfTrackVal()
    for i, data_batch in enumerate(all_defend_pc):
        pc = data_batch["cvar_res"]["pc"]
        perf.update(model.forward(pc)["logit"].detach().clone(), data_batch["label"])

    res = perf.agg()

    logger.info("defenced acc:")
    logger.info(res)

    return res
