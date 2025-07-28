import torch
import tqdm
from loguru import logger
from pytorch3d.structures import Pointclouds
from tqdm import tqdm

from src.attackers.hit_adv.backbone import HiT_ADV
from src.utils.model_eval import get_num_classes_by_dataset_name


def start_hit_adv_attack(model, device, loader_train, loader_test, num_batches_to_attack, args):
    logger.info(f"Start hit adv attack on {args.dataset} dataset, with {args.model} model")
    num_classes = get_num_classes_by_dataset_name(args.dataset)

    all_attacked_pc = []
    for batch_idx, data_item in enumerate(tqdm(loader_test)):
        if batch_idx >= num_batches_to_attack:
            break

        clean_input_item = {}
        pc = data_item["pc"].to(device).float()
        label = data_item["label"].to(device)

        target_label = (label + 1) % num_classes

        hit_adv = HiT_ADV(model, device)
        # hit adv需要法线信息
        pc_pyt3d = Pointclouds(points=pc)
        normals = pc_pyt3d.estimate_normals()

        pc_with_normals = torch.cat([pc, normals], dim=-1)

        adv_pc, success_num = hit_adv.attack(pc_with_normals.float(), label.long())

        all_attacked_pc.append({
            "real_data": torch.tensor(pc).cpu(),
            "attacked_data": torch.tensor(adv_pc).cpu(),
            "real_label": torch.tensor(label).cpu(),
            "target_label": torch.tensor(target_label).cpu(),
            "success_num": success_num,
        })

    return all_attacked_pc
