import torch
import tqdm
from loguru import logger
from pytorch3d.structures import Pointclouds
from tqdm import tqdm

from src.utils.model_eval import get_num_classes_by_dataset_name
from .backbone import build_gsda


def start_gsda_adv_attack(
    model, device, loader_train, loader_test, num_batches_to_attack, args
):
    logger.info(f"Start gsda attack on {args.dataset} dataset, with {args.model} model")
    num_classes = get_num_classes_by_dataset_name(args.dataset)

    all_attacked_pc = []
    for batch_idx, data_item in enumerate(tqdm(loader_test)):

        if batch_idx >= num_batches_to_attack:
            break

        pc = data_item["pc"].to(device).float()
        label = data_item["label"].to(device)
        target_label = (label + 1) % num_classes

        # gsda 需要法线信息
        pc_pyt3d = Pointclouds(points=pc)
        normals = pc_pyt3d.estimate_normals()

        gsda = build_gsda(model, num_classes)

        adv_pc, target, _ = gsda.forward(
            points=torch.cat([pc, normals], dim=-1),
            target=label.long(),
        )

        success_num = torch.sum(target == label)

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
