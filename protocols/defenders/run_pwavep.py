import argparse
import os
import sys

import torch
import torch_geometric
from loguru import logger
from torch.utils.data import DataLoader

torch_geometric.seed_everything(0)
project_root_path = os.path.dirname(os.path.abspath(__file__))
while True:
    parent_path = os.path.dirname(project_root_path)
    basename = os.path.basename(project_root_path)
    print("project root path: ", project_root_path)
    if basename == "pwavep" or project_root_path == "/":
        break
    project_root_path = parent_path
print("auto detect project root path: ", project_root_path)
sys.path.append(project_root_path)
sys.path.append(os.path.join(project_root_path, "src"))

if __name__ == "__main__":
    from src.models.trainer import load_model
    from src.utils.auto_gpu_chose import AutoGPUChoseManager
    from src.models.evaluator import validate_adv_examples
    from src.datasets.attacked_data import AttackedData
    from src.defenders.wavelet_def.starter import start_wavelet_gard_wrt_coff_energy

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attacker",
        type=str,
        default="eidos",
        help="attacker name",
        choices=["gsda", "eidos", "geoa3", "hit_adv", "si_adv"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pointnet",
        help="model name",
        choices=["pointnet", "dgcnn", "curvenet", "pointnet2"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ModelNet40",
        help="dataset name",
        choices=["ModelNet40", "ShapeNetPart"],
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size in training"
    )
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    args.model = args.model.lower()

    device = (
        f"cuda:{AutoGPUChoseManager().auto_choice()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    data_root = os.path.join(project_root_path, "data")
    ckpt_root = os.path.join(project_root_path, "ckpt")

    model = load_model(
        device,
        args.model,
        args.dataset,
        ckpt_root,
    )

    attacked_data = torch.load(
        f"{project_root_path}/attacked_data/{args.attacker}/{args.dataset}_{args.model}_res.pt",
        map_location=device,
    )

    clean_acc, asr = validate_adv_examples(attacked_data, model, device)

    logger.info(
        f"{args.attacker} at model {args.model} and dataset {args.dataset} attack result:"
    )
    logger.info(f"clean acc: {clean_acc['acc']}")
    logger.info(f"adv acc: {asr['acc']}")

    attack_loader = DataLoader(
        AttackedData(attacked_data),
        batch_size=args.batch_size,
        shuffle=False,
    )

    def_res = start_wavelet_gard_wrt_coff_energy(
        loader=attack_loader,
        device=device,
        model=model,
    )

    def_acc = round(def_res["acc"] * 100, 2)

    logger.info(f"{args.attacker} {args.model} {args.dataset}")
    logger.info(f"wavelet def result: {def_acc}")
