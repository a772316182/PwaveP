import torch
import tqdm
from loguru import logger
from tqdm import tqdm

from src.utils.model_eval import get_num_classes_by_dataset_name
from .backbone import siadv_attack


def start_si_adv_attack(
    model, device, loader_train, loader_test, num_batches_to_attack, args
):
    logger.info(
        f"Start SI adv attack on {args.dataset} dataset, with {args.model} model"
    )
    num_classes = get_num_classes_by_dataset_name(args.dataset)

    all_attacked_pc = []



    for batch_idx, data_item in enumerate(tqdm(loader_test)):
        if batch_idx >= num_batches_to_attack:
            break

        pc = data_item["pc"].to(device).float().clone()
        label = data_item["label"].to(device).clone()
        target_label = (label + 1) % num_classes

        total_size = len(pc)
        success_num = 0
        adv_pc = []
        for i in tqdm(range(len(pc))):
            hit_adv = siadv_attack(
                eps=0.16,
                step_size=0.007,
                max_steps=100,
                classifier=model,
                pre_head=torch.nn.Identity(),
                num_class=num_classes,
                top5_attack=False,
            )
            pc_item = torch.unsqueeze(pc[i], dim=0)
            label_item = torch.unsqueeze(label[i], dim=0)
            target_label_item = torch.unsqueeze(target_label[i], dim=0)
            adv_pc_item, _, is_success = hit_adv.forward(
                pc_item.float(), label_item.long()
            )

            adv_pc_item = torch.squeeze(adv_pc_item)

            adv_pc.append(adv_pc_item)
            success_num += int(is_success)

        adv_pc = torch.stack(adv_pc, dim=0)

        all_attacked_pc.append(
            {
                "real_data": torch.tensor(pc).cpu(),
                "attacked_data": torch.tensor(adv_pc).cpu(),
                "real_label": torch.tensor(label).cpu(),
                "target_label": torch.tensor(target_label).cpu(),
                "success_num": success_num,
            }
        )

    return all_attacked_pc
