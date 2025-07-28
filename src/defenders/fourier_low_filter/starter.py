import copy

import tqdm
from loguru import logger
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .filter import low_filter
from ...utils.model_eval import PerfTrackVal


def start_gsp_low_filter(
        victim_loader: DataLoader,
        device: str,
        model: Module,
        cutoff: float = 0.6695425186640739,
):
    logger.info("Starting GSP low pass filter")
    all_defend_pc = []
    for i, data_batch in enumerate(tqdm(victim_loader)):
        victim_pc = copy.deepcopy(data_batch["attacked_data"]).float().to(device)
        test_label = data_batch["real_label"]
        filter_res = low_filter(victim_pc, cutoff=cutoff)
        clean_input_item = dict()
        clean_input_item["pc"] = filter_res
        clean_input_item["label"] = test_label.detach().clone()

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
