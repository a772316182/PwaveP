import copy

import numpy as np
import torch
import tqdm
from loguru import logger
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .SOR import SORDefense
from ...utils.distance_utils import eval_cd
from ...utils.model_eval import PerfTrackVal


def start_sor(
    victim_loader: DataLoader,
    device: str,
    model: Module,
    sor_k: int = 3,
    sor_alpha: float = 1.1,
    key_clean_pc: str = "real_data",
    key_adv_pc: str = "attacked_data",
    key_clean_label: str = "real_label",
    key_adv_label: str = "target_label",
):
    defense_module = SORDefense(k=sor_k, alpha=sor_alpha)
    all_defend_pc = []

    for i, data_item in enumerate(tqdm(victim_loader)):
        batched_adv_pc = data_item[key_adv_pc].to(device).float()
        batched_clean_pc = data_item[key_clean_pc].to(device).float()
        adv_label = torch.squeeze(data_item[key_adv_label]).to(device)
        clean_label = torch.squeeze(data_item[key_clean_label]).to(device)

        victim_pc = copy.deepcopy(data_item["attacked_data"]).float().to(device)
        test_label = data_item["real_label"]
        sor_res = defense_module.forward(victim_pc)

        clean_input_item = dict()

        """
        sor processed results have different number of points in each
        thus need to reshape them to the same size by padding 0
        """
        sizes = [item.shape[0] for item in sor_res]
        variance = np.var(sizes)
        if variance > 0:
            sor_defend_batch_pc = [pc.detach().cpu().numpy() for pc in sor_res]
            # reshape all points to max size by padding 0
            max_size = np.max([pc.shape[0] for pc in sor_defend_batch_pc])
            defend_batch_pc = []
            for pc in sor_defend_batch_pc:
                new_pc = np.zeros((max_size, 3))
                new_pc[: pc.shape[0], :] = pc
                new_pc = new_pc.astype(np.float32)
                defend_batch_pc.append(new_pc)
            clean_input_item["pc"] = defend_batch_pc
        else:
            clean_input_item["pc"] = [
                item.detach().cpu().numpy().astype(np.float32) for item in sor_res
            ]

        clean_input_item["label"] = test_label.detach().clone()
        clean_input_item["pc"] = torch.tensor(clean_input_item["pc"], device=device)

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
