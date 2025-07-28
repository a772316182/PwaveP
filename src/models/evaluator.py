import torch
import tqdm
from tqdm import tqdm

from src.utils.model_eval import PerfTrackVal


@torch.no_grad()
def validate(loader, model, device, break_on=10000):
    perf = PerfTrackVal()

    for i, data_batch in enumerate(tqdm(loader, total=len(loader))):
        if i >= break_on:
            break
        pc = data_batch["pc"]
        label = data_batch["label"]

        pc = pc.to(device)
        pred = model.forward(pc)
        perf.update(logits=pred["logit"], ground_truth_label=label)

    return perf.agg()

@torch.no_grad()
def validate_adv_examples(loader, model, device):
    adv_perf = PerfTrackVal()
    clean_perf = PerfTrackVal()

    for i, data_batch in enumerate(tqdm(loader, total=len(loader))):
        clean_pc = data_batch["real_data"].to(device).float()
        clean_label = data_batch["real_label"].to(device).long()
        adv_pc = data_batch["attacked_data"].to(device).float()

        clean_logit = model.forward(clean_pc)["logit"]
        adv_logit = model.forward(adv_pc)["logit"]

        clean_perf.update(logits=clean_logit, ground_truth_label=clean_label)
        adv_perf.update(logits=adv_logit, ground_truth_label=clean_label)

    return clean_perf.agg(), adv_perf.agg()