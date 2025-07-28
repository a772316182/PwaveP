import torch
import tqdm
from loguru import logger
from pytorch3d.structures import Pointclouds
from tqdm import tqdm

from src.utils.model_eval import get_num_classes_by_dataset_name
from .backbone import build_geoa3_attack


def start_geoa3_attack(
    model, device, loader_train, loader_test, num_batches_to_attack, args
):
    logger.info(f"Start geoa3 attack on {args.dataset} dataset, with {args.model} model")
    num_classes = get_num_classes_by_dataset_name(args.dataset)

    all_attacked_pc = []
    for batch_idx, data_item in enumerate(tqdm(loader_test, desc="Attacking")):

        if batch_idx >= num_batches_to_attack:
            break

        clean_input_item = {}
        pc = data_item["pc"].to(device).float()
        label = data_item["label"].to(device)

        target_label = (label + 1) % num_classes

        # geoa3 需要法线信息
        pc_pyt3d = Pointclouds(points=pc)
        normals = pc_pyt3d.estimate_normals()

        geoa3 = build_geoa3_attack(model, num_classes)

        adv_pc, target, _ = geoa3.forward(
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
